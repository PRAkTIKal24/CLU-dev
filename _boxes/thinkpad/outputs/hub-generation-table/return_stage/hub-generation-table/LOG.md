# hub/generation-table — RUNNING LOG
**Agent:** `autonomous-hub` · **Branch:** `hub/generation-table` · **Worktree:** `C:/Users/PratikJawahar/chlu/CHLU-hub-generation-table`
**Machine:** FORGIS_14, Windows 11 Pro AMD64, 16 cores / 63 GB / Intel Arc Pro (⛔ no CUDA). Core budget **5 threads**.
**Bundle:** `chlu-bundle-v2`, pin `4125fce70b60ba1d9627ddc352d25bad5a08974a`, HEAD `8e36bc9`, tree `345244bb…` == `pin_tree`.
**START_UTC:** `2026-09-16T19:10:50Z` → **12 h deadline `2026-09-17T07:10:50Z`** (COMMON §1b head ruling, supersedes the task file's 10 h).

---

## 2026-09-16T19:10Z — BOOT

- Read, in order: `LAUNCH-COMMON.md`, `LAUNCH-hub-generation-table.md`, `.claude/autonomous-hub-charter.md`,
  `.claude/AGENT_PROTOCOL.md`, `.claude/handover_context.md` §7, `.claude/tasks/hub-generation-table.md`. ⛔ Nothing else
  from the campaign archive.
- **Override quoted verbatim, as required by COMMON §0:**
  > **Override recorded (Hub, 2026-09-16): launch `hub/refusal`, `hub/composition` and `hub/generation-table` on v2
  > despite `test_documents_are_decoded_as_utf8_regardless_of_the_platform_locale`.** The failure is in the test's
  > `_WindowsFlavourPath` mock, not the bundle. The mock's `read_text()` calls `read_bytes().decode()`, which does
  > **not** fold universal newlines, while real `Path.read_text()` opens in text mode and does. On Windows the fixture
  > is written by a real `write_text()`, which expands `\n → \r\n`, so the mock counts two characters per newline and
  > `chars` exceeds `len(text)` by exactly the newline count. **Nothing to do with encoding.** It passes on macOS only
  > because macOS does not expand newlines. The mock is fixed on the Mac and ships in the next cut; the UTF-8
  > behaviour it guards is unaffected and still enforced by the shipped code's explicit `encoding="utf-8"`.
  - ⛔ I treat this as covering **exactly that one test**. Any other red check is information, not something to wave through.
- Worktree created from **HEAD**, ⛔ not `ae907466` (COMMON §1a: `ae907466` does not exist in this repo; one commit only):
  `git worktree add /c/Users/PratikJawahar/chlu/CHLU-hub-generation-table -b hub/generation-table HEAD` → OK at `8e36bc9`.
- Env: `PYTHONUTF8=1`, `PYTHONPATH=<worktree>`, `PY=$CHLU_ROOT/.venv/Scripts/python.exe` (3.11.16, jax 0.9.0, numpy 2.4.1),
  `OMP/MKL/OPENBLAS_NUM_THREADS=5`. Import check: `chlu.__file__` resolves **into my worktree**, not the bundle. ✅
- Reusable shell preamble written to `$CHLU_OUT/hubenv.sh` so every shell in this run is identical.
- ⛔ **Scored-run budget UNKNOWN (COMMON §1c).** The Head's binding reading is 3 **per target across every machine**, and
  this box cannot see the Mac's count. **I will not spend a scored run until a human tells me how many of the three
  remain for `generation-table`.** Question raised to the operator below. Reading, planning, PREREG and val/dry work
  continue meanwhile — all unrestricted.

### ⚠ OPEN QUESTION TO THE OPERATOR (raised 19:10Z, blocking only the scored runs)
> **How many of the 3 scored runs for target `generation-table` remain unspent?** I have spent 0 on this box. If the Mac
> has already scored against held-out A/B/C for this target, I need that count before my first scored run.
> Until answered I will treat the budget as **unknown** and do everything else.

### My starting position, restated so I cannot drift off it
- ⛔ I start from **"is the store earning its bytes"**, ⛔ not from "beat 0.5949". At matched bytes the store is
  **measured DECORATIVE** today. Clearing that is a **precondition, not an achievement**.
- ⭐ Honest dynamic range is **~0.33 → 0.865**, not 0 → 0.865. The constant-output floor (2.67 ± 0.47 distinct tuples of 72)
  already scores **0.3272 ± 0.0633** and **ties `lace` 0.3117**. A mid-0.4 is ~0.07 above free. ⛔ Not progress.
- ⭐⭐ **Every aggregate I quote carries its distinct-tuple count on the same line. No exceptions.**
- ⛔ `x0` 0.6852 is at **8× bytes (4,194,304 B)**; at matched bytes (524,288 B) it is **0.5008** and loses to `clu` 0.5949.
- ⭐ Ceiling `knn_proto` **0.8650 ± 0.0122** at 475,136 B, rendering **56.00 ± 3.74 / 72** (⛔ not 72/72). `m = 0.020`.
- ⛔ Inherited and NOT to be re-derived: dynamics are the defect (0.6852 launch → 0.2114 settled @2000 steps);
  steering is real but small (β=0 → 1 distinct tuple/72, 0.0000 at rung B); placement is a null (inherited ward/corr
  grouping already beats the D-matched random floor 50×); **the codebook is the largest identified loss —
  90.5 % of `scale`, 100 % of `orientation`**, partition 9.5 % / 0 %.
- ⇒ **Working hypothesis at boot: the codebook is the lever.** It is the only signposted opportunity of that size, and
  it is explicitly inside charter §B's "everything else may change" list.

**Next:** map the frozen harness (`exp_gen_cell`, `gen_store`, byte ledger, bypass assert, external classifier) and
locate the codebook path, read-only, before touching anything.

## 2026-09-16T19:13Z — MAP OF THE MECHANISM (read-only; ⛔ nothing edited)

**The store, exactly.** `chlu/core/gen_store.GenStore` is **product quantisation**, nothing more exotic:
`K` wells **partition** the `D` feature channels; well `k` owns `d_k` channels and holds `M` learned modes of
size `d_k`. From `gen-store-grouping/runs/shapes3d_recon/store_plan.json` (pinned, R0 rung):

| | |
|---|---|
| `D` (feat_shape `128×4×4`) | **2048** |
| `K` wells (ward on channel correlation) | **5**, sizes `[303, 305, 573, 501, 366]`, 71 dead channels |
| `M` modes per well | **64** |
| `store_bytes` | `M·D·4` = **524,288 B** ✅ matches `expect_deployed_store_bytes` |
| `bits_per_read` | `Σ_k log2(M_k)` = `5·6` = **30.0** |

⭐⭐ **THE STRUCTURAL FACT THAT MATTERS MOST, and it is in the shipped docstring:**
`GenStore` — *"the wells partition the coordinates, so `sum_k d_k = D` and `store_bytes = M·D·4`
**independently of K**"*. ⇒ **K is FREE in bytes.** Going `K=5 → K=20` at `M=64` keeps the ledger at exactly
524,288 B and takes the channel from 30 bits to 120, and the reachable read set from `64^5` to `64^20`.
No assertion anywhere binds `bits_per_read` — `assert_no_bypass` only *records* it; only
`expect_deployed_store_bytes` is enforced. Charter §B lists **partition** and **codebook** as changeable.

**The read path, exactly.**
`requests_from_labels` → `gen_dream.address_requests` → `gen_compose.request_posterior` (softmax) →
`compose_codes` (**argmax**, one mode per well) → `gen_store.read_from_codes` (gather + scatter) → decoder → classifier.

⭐⭐ **THE DEFECT I THINK I HAVE FOUND — `fit_address_book`/`address_requests` is NAIVE BAYES.**
`chlu/core/gen_dream.py:296–339`. Per well `k`, the score for mode `m` is
`log prior_k(m) + Σ_f [ log P(m | factor_f = v_f) − log prior_k(m) ]`
— a log-linear pool of **per-factor marginals**, i.e. factors assumed conditionally independent given the mode.
Each factor is fitted **alone** (`cnt = bincount(C[Y[:,f]==v, k])`). ⇒ the address book has **no term that can
represent a conjunction**, which is precisely what a composition request *is*. This is a plausible mechanical
cause of the measured collapse to **9.00 ± 0.82 distinct tuples of 72** on the `clu` row, and it is
*consistent with* the inherited attribution ("the codebook costs 90.5 % of `scale`, 100 % of `orientation`"):
under naive-Bayes pooling a well is captured by whichever factor has the sharpest per-factor marginal in it,
and the weaker factors (scale, orientation) contribute a near-flat term that the argmax discards.
⚠ **Hypothesis, not yet measured.** It is the first thing I will measure, on dev data.

**Two levers, both inside charter §B, both byte-free:**
- **L1 — finer partition (`K↑` at fixed `M`).** Zero byte cost (proved above). Classic product quantisation:
  distortion falls monotonically with the number of sub-quantisers at fixed `k·D` memory. `K=5` is extremely
  coarse by PQ standards.
- **L2 — a conjunction-capable address book.** Replace naive-Bayes pooling with something that can score a
  *joint* assignment. ⚠ Must stay inside §B.6: labels may ground request words **at the interface only**
  (the address book **is** the interface) and ⛔ never in the landscape, the read or the write. The modes
  themselves must stay label-free.

⛔ **Neither lever is "shrinking the store's role".** Both make the store carry *more*, not less. I will check
that explicitly with `store_removed` on every dev number.

**Environment work done (both sanctioned, both additive, ⛔ nothing pinned touched):**
- `h5py 3.16.0` installed into the bundle venv (`uv pip install --python <venv> h5py`). It is a **declared
  optional extra** in `pyproject.toml` (`[project.optional-dependencies] gen`), needed by
  `chlu/data/shapes3d.load_labels`. ✅ numpy stayed **2.4.1** — no resolver drift.
- `3dshapes.h5` fetched to `$CHLU_GEN_DATA_ROOT/shapes3d/` by a resumable unit
  (`$CHLU_OUT/dl/get_h5.sh`, ledger at `$CHLU_OUT/dl/ledger.txt`). **SHA-256 verified against the shipped
  constant: `0a0f6ed9…be1785` — MATCH.** Download + verify took **129 s**. COMMON §1a names this as the
  command the second machine needs; the raw set is deliberately never bundled. The unit refuses to overwrite
  an existing published file, because three hubs share this data root.

**Feature caches on disk:** seed 0 `scope=full`, 480,000 rows, `D=2048`, float16, sha `d6728f01…`;
seeds 1 and 2 `scope=used`, 9,288 rows each. ⛔ Three **different** encoders (three different
`encoder_param_hash`), as intended — the seed is the whole stack's seed.
⚠ **COMMON §4 binds here: I will not re-encode the feature cache.** If a mechanism of mine needs rows outside
a seed's `used` cache, I stop and report rather than extend it.

## 2026-09-16T21:16Z — ⭐⭐ FINDING 1 (structural, arithmetic, ⛔ NOT a scored run): the free floor is **0.5278**, not 0.3272

I computed the **label structure of the B row set**. This uses no arm, no model, no classifier output and no
measured accuracy — only the Montero predicate, which is part of the split definition. `eval_b` is exactly
`split.test_rows`, and it is a **complete Cartesian product**:

| factor | values realised in B | count each | modal fraction |
|---|---|---|---|
| floor_hue | 2 (`{8,9}`) | 36 | 0.5000 |
| wall_hue | 2 (`{8,9}`) | 36 | 0.5000 |
| object_hue | 2 (`{8,9}`) | 36 | 0.5000 |
| scale | 3 (`{5,6,7}`) | 24 | 0.3333 |
| **shape** | **1 (`{1}` = cylinder)** | **72** | **1.0000** |
| orientation | 3 (`{12,13,14}`) | 24 | 0.3333 |

composition-B aggregate is the **unweighted mean of six per-factor top-1 accuracies**
(`chlu/eval/gen_probe.probe_accuracy`). A **single constant output** therefore scores, at best,
`mean(1/2, 1/2, 1/2, 1/3, 1, 1/3)` =

> ## ⭐⭐ **0.5278, with 1.00 distinct factor tuples of 72.**

⛔ **This is strictly stronger than the trap my task file warned me about**, and it moves in the direction that
hurts me most, so I am recording it before I have any result of my own:

| row | composition-B | distinct tuples / 72 | deployed bytes |
|---|---|---|---|
| `store_removed` (the null) | 0.1204 ± 0.0131 | 1.00 | 524,288 |
| `ebm_compose` | 0.1937 ± 0.0182 | 72.00 | 14,226,660 |
| `dynread` B (it lost) | 0.2114 ± 0.1033 | 16.0 | 8,388,608 |
| `lace` (CELL instrument) | 0.3117 ± 0.0153 | 71.67 | not byte-matched |
| measured near-constant (`x0/A_row_lookup`) | 0.3272 ± 0.0633 | 2.67 ± 0.47 | — |
| atom-mean lookup, matched bytes (§C floor) | 0.3927 ± 0.0986 | — | 524,288 |
| `x0` addressed read, matched bytes | 0.5008 ± 0.0498 | — | 524,288 |
| ⭐⭐ **BEST CONSTANT OUTPUT (structural bound, this finding)** | **0.5278** | **1.00** | **0** |
| `clu` codebook read | 0.5949 ± 0.0082 | 9.00 ± 0.82 | 524,288 |
| `x0` addressed read, **8× bytes** | 0.6852 ± 0.0173 | — | 4,194,304 |
| `knn_proto` — the ceiling | 0.8650 ± 0.0122 | 56.00 ± 3.74 | 475,136 |

**What this means, stated plainly:**
- ⛔ **`x0` at matched bytes (0.5008) is BELOW a constant output.** So is the atom-mean lookup, and so is `lace`,
  `ebm_compose`, `dynread` and `store_removed`. **Six of the ten rows lose to emitting the same image 72 times.**
- ⭐ `clu` 0.5949 is **+0.0671** over the best constant — about **20 %** of the way across the honest band.
- ⭐ **The honest dynamic range is `0.5278 → 0.8650`, a width of 0.337** — ⛔ not `0.33 → 0.865` (0.54 wide) and
  emphatically not `0 → 0.865`.
- ⇒ ⛔ **A "0.62" would be 0.09 above free and I must never present it as a win.** Recorded now, before I have a number.
- ⚠ The measured 0.3272 near-constant row is *not wrong*: it is what one **particular** constant scores. 0.5278
  is what the **best** constant scores. The gap between them is entirely "which constant", and nothing in the
  metric stops an arm from landing on the good one.
- ⚠ **Achievability caveat, to be measured, not assumed:** 0.5278 requires a constant image the external
  classifier reads as (hue 8-or-9 ×3, scale 5/6/7, **cylinder**, orientation 12/13/14). Whether any decodable
  feature vector renders to that tuple is an empirical question. I will measure it **on DEV**, never on B.

**Why this is not a scored run.** It is arithmetic on `split.test_rows`' factor labels — the definition of the
held-out cell, not a measurement through it. No image was rendered, no classifier was called, no arm was run.
I record it as **scored runs spent: 0/?**.

## 2026-09-16T21:18Z — DEV protocol (mechanism selection happens here, ⛔ never on B)

`exp_gen_cell.run_seed` computes composition-B the moment it runs, so ⛔ **I do not run it at all** until a
pre-registered scored run. Instead:

- **DEV corner** = a cell with **Montero's exact shape** (2-value band on each of 3 hues · 3-value corner of
  scale · exactly one shape · 3-value corner of orientation → **72 rows**) built at a **different location**
  inside `split.train_rows`. 50 such corners enumerated; each overlaps the 32,768-row store pool in only
  **2–10 rows**, which I **remove from the pool and from the address-book fit**, so the DEV corner is held out
  from the store exactly as B is.
- ⚠ **Declared limitation of DEV, stated up front:** the frozen **encoder and decoder** were fitted on Montero's
  train split, which *contains* the DEV corner. So DEV is **optimistic in absolute level**. ⭐ It is used only to
  **rank mechanisms**, and the bias applies identically to every arm I compare on it.
- ⭐ **The comparison that will count is run IN ONE PROCESS with the rivals**, because the rivals' published rows
  are Mac-produced and mine are Windows-produced (COMMON §4). `run_seed` already scores every arm against the
  same gathered features in the same process, and `regression_check` reports any rival that lands below its
  published constant. ⇒ **my like-for-like claim will be against my own run's rival rows**, with the published
  ones quoted beside them as the regression check.

## 2026-09-16T19:19Z — the addressing head, re-estimated at **ZERO byte cost** (`chlu/core/gen_address.py`, new file on my branch)

The shipped head's forward pass is `score[n,k,m] = b[k,m] + Σ_f W[offset_f + r[n,f], k, m]`
(`gen_dream.address_requests`). Its parameters are fitted by **naive-Bayes counting** — each factor's block
alone, from `P(mode | factor_f = v)`. ⭐ **But that functional form is just a multinomial logistic regression,
and naive Bayes is one particular closed-form setting of its weights.** So the same tensor can be fitted by
**maximum conditional likelihood** instead, and the head's size does not move by one byte.

`chlu/core/gen_address.py::fit_discriminative_address_book` does exactly that: per-well multinomial logistic
regression on the one-hot factor design, Adam, ridge on `W` only, **initialised at the naive-Bayes estimate**.

**Verified now, not asserted** (`$CHLU_OUT/dev/test_head_equivalence.py`):

```
steps=0 log_prior bit-identical : True
steps=0 log_cond  bit-identical : True
posterior max |delta|           : 0.0
address head bytes  naive-Bayes : 74240 (K*M + sum_f V_f*K*M) * 4
address head bytes  discrim.    : 74240
BYTES UNCHANGED                 : True
after 30 steps, bytes           : 74240 unchanged: True
RESULT: PASS
```
⭐ **74,240 B is exactly `GenCellConfig.addressing_head_bytes_reference`.** ⛔ The head does not grow.
⭐ `steps=0` is **bit-identical** to the shipped estimator, so a null result here cannot be an implementation
difference — the new estimator is a strict superset of the old one and starts from it.
⭐ Charter §B.6 holds unchanged: labels enter the **address book only**; the fitting targets are
`address_codes(store, X_store)` — the store's own codes for its own features — and the read is still
`read_from_codes(store, codes)`, so `assert_no_bypass`'s signature check passes untouched.
⛔ `chlu/experiments/exp_gen_cell.py` and every other frozen-harness file are **UNTOUCHED** (charter §A.4).
My arm will be scored by *calling* the harness's own `render` / `score_images` / `distinct_counts` /
`load_external_classifier`, the way `dynread` was wired — ⛔ never by a second implementation.

**Running now (DEV, ⛔ unscored):**
- `exp01` — reproduce the shipped CLU read on the DEV corner + `store_removed` + decoder ceiling + the
  **realised** best-constant (does a real constant actually achieve the 0.5278-style structural bound?).
- `exp02` — ⭐ **the diagnostic that decides where I spend the next 10 hours**: sweep `K ∈ {5,10,20,40,64}` at
  `M = 64` (⇒ `store_bytes = M·D·4 = 524,288 B` for **every** K), and at each K measure both
  - `quant_ceiling` — codes taken from the FEATURES by nearest mode. ⛔ A **diagnostic only** — it bypasses
    the request and can never be reported as an arm. It is the best *any* addresser could do through that codebook.
  - `clu` — codes from the REQUEST through the shipped naive-Bayes head.
  ⇒ `clu ≪ quant_ceiling` means the defect is **ADDRESSING**; a low `quant_ceiling` means it is the **CODEBOOK**.

## 2026-09-16T19:25Z — ⭐⭐ DEV-01 RESULT: the 0.5278 bound is **REALISED**, and the shipped read is **BELOW IT**

DEV corner: `hue_lo=0.2 · scale_hi · shape=cube(0) · orientation_hi` — Montero's exact shape, 72 rows,
held out of the store pool (9 rows removed) and out of the address book. Seed 0. `K=5, M=64, 524,288 B,
30.0 bits/read`. Wall clock 636 s. ⛔ DEV numbers — **not** B, **not** a scored run.

| DEV arm | composition aggregate | **distinct factor tuples / 72** | floor | wall | object | scale | shape | orient |
|---|---|---|---|---|---|---|---|---|
| `store_removed` (laundering control) | **0.2222** | **1** | 0.00 | 0.50 | 0.50 | 0.33 | 0.00 | 0.00 |
| ⭐ **`clu`, the shipped read** | **0.4722** | **9** | 1.00 | 1.00 | 0.75 | **0.00** | **0.00** | **0.08** |
| ⭐⭐ **best constant, REALISED** | **0.5278** | **1** | 0.50 | 0.50 | 0.50 | 0.33 | **1.00** | 0.33 |
| decoder ceiling (real features) | **0.9630** | **72** | 1.00 | 1.00 | 1.00 | 1.00 | 0.78 | 1.00 |

**Three things this settles.**

1. ⭐⭐ **The best-constant bound is ACHIEVABLE, not just arithmetic.** I emitted ONE real corner feature 72
   times through the frozen decoder and the frozen classifier. It scored **0.5278 — to the last digit the
   structural bound** — with per-factor accuracies exactly the modal fractions `(½,½,½,⅓,1,⅓)`.
   ⇒ **FINDING 1 is confirmed empirically.** ⛔ Any arm at or below ~0.53 is doing no better than emitting the
   same picture 72 times.
2. ⛔⛔ **On DEV the shipped `clu` read (0.4722, 9 tuples) LOSES to a constant output (0.5278, 1 tuple).**
   It "wins" the hue columns outright (1.00 / 1.00 / 0.75 vs 0.50 / 0.50 / 0.50) and then throws the whole
   lead away on **scale 0.00, shape 0.00, orientation 0.08**. ⚠ On B the published `clu` is 0.5949, i.e.
   **+0.067 over the constant**, and B's `shape` column is free (one class, cylinder) whereas my DEV corner's
   is not — so the DEV corner is a **harder** instrument on exactly the axis that matters. Good: I want the
   harder one for selection.
3. ⭐ The inherited attribution reproduces **exactly**: the codebook read keeps hue and destroys
   **scale / shape / orientation**. ⛔ I did not re-derive it to confirm it; it fell out of the first DEV run.

**The honest DEV band is `0.5278 → 0.9630`.** ⭐ Every DEV number I quote from here carries its distinct-tuple
count and is read against **0.5278**, not against 0.

⚠ **Cost note, for the ledger:** 470 s of those 636 s were a badly-shaped loop (I rendered 72 identical images
72 times to find the best constant, when one render of the 72 dev rows gives every candidate's prediction).
Fixed in the next experiment. ⛔ Recorded rather than quietly dropped.

- ⚠ **Correction to the DEV-01 entry above (new bullet, ⛔ the entry is not rewritten):** the DEV corner
  removes **2** rows from the 32,768-row store pool, not 9. Realised pool = **32,766**. The 9 was a different
  corner in the enumeration table. Nothing else in the entry changes.

## 2026-09-16T19:40Z — ⭐⭐⭐ DEV-02, K=5: **THE CODEBOOK IS THE WHOLE DEFECT.** Addressing has ≤ 0.016 of headroom.

First point of the K-sweep, seed 0, same DEV corner, `M=64`, `store_bytes = 524,288 B`, `30.0 bits/read`:

| DEV row (K=5) | aggregate | **distinct tuples / 72** | what it is |
|---|---|---|---|
| `store_removed` | 0.2222 | 1 | laundering control |
| `clu` (naive-Bayes addresser) | **0.4722** | **9** (10 distinct code configs) | the shipped arm |
| ⭐ **`quant_ceiling`** | **0.4884** | **11** | ⛔ **DIAGNOSTIC ONLY** — codes taken from the TRUE features by nearest mode. The best **any** addresser could do through this codebook. |
| best constant (realised) | 0.5278 | 1 | — |
| decoder ceiling | 0.9630 | 72 | — |

`code_agreement` (request's code == the feature's own nearest-mode code) = **0.872**.

### ⛔⛔ What this kills, immediately

> **The shipped naive-Bayes addresser is within 0.0162 of the absolute ceiling of the codebook it addresses.**

⇒ ⛔ **Fixing the address book at `K=5` is worth at most +0.016.** My whole "naive Bayes cannot represent a
conjunction" hypothesis is *correct as a description* and **almost worthless as a lever** — the addresser is
already extracting 96.7 % of what this codebook can give. ⭐ I am recording this as a **rejected hypothesis of
my own**, four hours before I could have spent the run on it.

> ⭐⭐ **And the ceiling itself is BELOW the free floor: `quant_ceiling` 0.4884 < best constant 0.5278.**
> A `K=5, M=64` product codebook **cannot represent more than 11 distinct factor tuples of 72 even when the
> codes are read off the true features.** The 30-bit channel is not the constraint (30 bits names 10⁹ codes,
> and naming a Shapes3D tuple needs 19). ⛔ **The constraint is that the bits are spent quantising a 2048-dim
> feature vector in 5 blocks of ~400 dims, so they buy variance — which is colour — and not geometry.**

That is exactly the measured per-factor signature: hue 1.00 / 1.00 / 0.75, **scale 0.00, shape 0.00,
orientation 0.08**. ⭐ It also explains the inherited "the codebook costs 90.5 % of `scale` and 100 % of
`orientation`" without re-deriving it.

### ⇒ The lever is `K`, and §B allows it and the byte ledger does not move
`store_bytes = M·D·4` **independently of K** (shipped docstring, verified: 524,288 B at every K in this sweep).
⭐ `K` is the number of sub-quantisers in a product quantiser; distortion falls monotonically in `K` at fixed
`k·D` memory. `K=5` over `D=2048` is ~400 dims per 6-bit code — extraordinarily coarse.
**Sweep running:** `K ∈ {10, 20, 40, 64}`. ⚠ 874 s per K point on 5 threads.

### ⚠ The byte-honesty problem I am creating, named now before I have a number
The **addressing head** is `(K·M + Σ_f V_f·K·M)·4 = 232·K·M` B → `14,848·K`. At `K=5` that is the shipped
**74,240 B**; at `K=20` it is **296,960 B**. Deployed store stays 524,288 B, but *store + head* would go
598,528 → 821,248 B against `knn_proto`'s 475,136 B. ⛔ **That is not a match and I will not present it as one.**
Two byte-neutral repairs to test, both declared and measured, ⛔ neither adopted before it is checked:
- **int8 head with a per-well scale** — ¼ the bytes, so `K=20` lands at **74,240 B**, byte-identical to the
  shipped head. Must be shown not to move the number.
- **low-rank head** `b[k,m] + (E_{f,v}·U_k)[m]`, `E (ΣV_f × r)`, `U (K × r × M)` → `(57r + K·r·M)·4`;
  at `r=8, K=20` that is **42,784 B**, *smaller* than shipped.
⭐ Either way I will report **store bytes, head bytes and their total**, beside `knn_proto`'s 475,136 B,
in every table I write. ⛔ The shipped convention excludes the head from `deployed_bytes` — that convention
favours me, so I will not hide behind it.

## 2026-09-16T19:53Z — ⛔ **I WAS WRONG ABOUT `quant_ceiling`. Correcting it before it costs me anything.**

DEV-03, K=8, balanced partition, Lloyd k-means, `M=64`, store 524,288 B:

```
K=8 bits=48.0 store=524288 head=118784 | mse=0.0545 | ceil=0.4907(12) |
    nb=0.5116(11,18c) | disc=0.5324(12,18c) | sr=0.1389(1) | 588s
```

⛔ **`clu_nb` = 0.5116 is ABOVE `quant_ceiling` = 0.4907.** So `quant_ceiling` is **not a ceiling for this
metric**, and my 19:40 entry over-claimed. The correction, precisely:

- `quant_ceiling` takes the codes that minimise **feature-space distortion** (nearest mode to the true
  feature). composition-B is **classifier accuracy on the decoded image**, and the distortion-optimal code is
  *not* the accuracy-optimal code. A request-derived code can land on a codeword that decodes to a
  better-classified image than the nearest one does.
- ⇒ ⭐ **Rename in every later table: `quant_ceiling` → `nearest_mode_reference` (the distortion-optimal
  addresser).** ⛔ It is a reference, not a bound.
- ⇒ ⛔ **My 19:40 sentence "the shipped addresser is within 0.0162 of the absolute ceiling of the codebook,
  so fixing the address book is worth at most +0.016" DOES NOT FOLLOW and is withdrawn.** The K=8 row is the
  counter-example: the discriminative head beats the distortion-optimal addresser by **+0.042**.
- ⭐ What *does* survive from the 19:40 entry, unchanged: the codebook at `K=5` renders only **11 distinct
  factor tuples of 72** under the distortion-optimal addresser, `scale/shape/orientation` are destroyed,
  and `store_bytes` is invariant in `K`. Those were measurements, not inferences.

⭐ **And the first positive signal, at zero byte cost in the store:**

| DEV, K=8 | aggregate | distinct tuples / 72 | distinct code configs |
|---|---|---|---|
| `store_removed` | 0.1389 | 1 | — |
| nearest-mode reference | 0.4907 | 12 | — |
| `clu` + **shipped naive-Bayes head** | 0.5116 | 11 | 18 |
| ⭐ `clu` + **discriminative head (SAME 118,784 B)** | **0.5324** | **12** | 18 |
| best constant | 0.5278 | 1 | — |

⭐ **The discriminative head is worth +0.0208 over naive Bayes at identical head bytes** (0.5116 → 0.5324).
⛔ But 0.5324 is only **+0.005 over a constant output**, so this is a *mechanism* signal, ⛔ **not a result**.
⚠ And the head at K=8 is **118,784 B**, not the shipped 74,240 B — declared, and to be repaired later.

**Engineering:** `fit_discriminative_address_book` took **479 s** of the 588 s — `np.add.at` on the one-hot
gradient. Rewriting it as two sparse mat-muls, vectorised over wells. exp03 stopped (K=8 is on disk and the
ledger resumes); it will be restarted on the fast fitter.

## 2026-09-16T20:21Z — ⭐⭐⭐ DEV-03: **K IS THE LEVER**, and it costs the store nothing. But the HEAD grows with it.

Seed 0, same DEV corner, `M = 64` throughout ⇒ `store_bytes = M·D·4 = **524,288 B at every K**` (verified per row).
Balanced correlation partition (ward leaf order cut into equal blocks), Lloyd k-means codebook.
`nearest_mode_reference` = the distortion-optimal addresser (⛔ a reference, **not** a bound — see the 19:53 correction).

| K | bits/read | store B | **head B** | recon MSE | nearest-mode ref | `clu` + naive-Bayes head | `clu` + **discriminative** head | `store_removed` |
|---|---|---|---|---|---|---|---|---|
| **5** (shipped: ward + `train_store`) | 30 | 524,288 | **74,240** | — | 0.4884 (11) | **0.4722 (9)** | — | 0.2222 (1) |
| 8 | 48 | 524,288 | 118,784 | 0.0545 | 0.4907 (12) | 0.5116 (11) | **0.5324 (12)** | 0.1389 (1) |
| 16 | 96 | 524,288 | 237,568 | 0.0517 | 0.5463 (19) | 0.5625 (16) | **0.5856 (15)** | 0.0556 (1) |
| ⭐ **32** | 192 | 524,288 | 475,136 | 0.0446 | 0.7014 (30) | ⭐ **0.6921 (25)** | 0.6782 (32) | 0.1389 (1) |
| 64 | 384 | 524,288 | 950,272 | 0.0381 | 0.7130 (38) | 0.6713 (38) | 0.6528 (29) | 0.0556 (1) |

*(parenthesis = distinct factor tuples of 72 — ⭐ on every single number, without exception)*
DEV references on the same corner: **best constant 0.5278 (1 tuple)** · **decoder ceiling 0.9630 (72 tuples)**.

### What the sweep says
1. ⭐⭐ **`K = 5 → 32` moves the DEV arm from 0.4722 (9 tuples) to 0.6921 (25 tuples) with the store byte
   ledger unchanged at 524,288 B.** That is from **below** a constant output to **+0.164 above** it, and from
   9 distinct tuples to 25. The shipped `K = 5` is simply far too coarse a product quantiser for `D = 2048`
   (~400 dimensions per 6-bit code).
2. ⛔ **`K = 64` is past the peak for the ARM** even though the codebook keeps improving
   (reference 0.7014 → 0.7130, MSE 0.0446 → 0.0381): both heads *fall* (0.6921 → 0.6713 nb). ⇒ beyond
   `K ≈ 32` the bottleneck moves from the codebook to the **addresser**. At `K=64` every request gets its own
   code (72 distinct code configs) but only 38 distinct tuples survive the decoder — variety without content.
3. ⚠ **The discriminative head helps at `K ≤ 16` (+0.021 at K=8, +0.023 at K=16) and HURTS at `K ≥ 32`**
   (−0.014 at K=32, −0.019 at K=64) at a fixed `l2 = 1e-3`. ⇒ it is **over-fitting** as the head grows; the
   ridge has to scale with `K`. ⛔ I will not quote the discriminative head as a win until that is tuned on DEV.
4. `store_removed` collapses to **1 distinct tuple** at every K (0.056–0.222). ⭐ Falsifier 3 is holding so far.

### ⛔⛔ THE BYTE PROBLEM, STATED PLAINLY BEFORE I HAVE A HEADLINE
`head_bytes = (K·M + Σ_f V_f·K·M)·4 = 232·K·M`, and `M` is **pinned to 64 by the 524,288 B store budget**
(`store_bytes = M·D·4`). ⇒ **`head_bytes = 14,848·K`, linear in K, and no allocation trick removes it**
(a non-uniform allocation still has `Σ_k M_k = 64K`).

| arm | store B | head B | **TOTAL B** |
|---|---|---|---|
| `knn_proto` (the ceiling) | 0 — its prototype table **is** its head | 475,136 | **475,136** |
| shipped `clu`, K=5 | 524,288 | 74,240 | **598,528** |
| my K=32 with the FULL head | 524,288 | 475,136 | ⛔ **999,424** |

⭐ **Two things I will not do:** ⛔ hide behind the shipped convention that excludes the addressing head from
`deployed_bytes`, and ⛔ quote a `K=32` number against `knn_proto` at 999,424 B.
⭐ **The target I am setting myself instead, now, before measuring it:** `K = 32` with a head of **≤ 74,240 B**
— *byte-identical to the shipped `clu` arm in every column of the ledger*, differing only in `bits_per_read`
(192 vs 30). A rank-`r` factorisation `W[v,k,m] = Σ_r E[v,r]·U[k,r,m]` costs `(Σ_f V_f·r + K·M·r)·4`;
at `K=32, M=64` that is `8,420·r` B, so **`r = 8` gives 67,360 B ≤ 74,240 B**. That is the configuration to build.
⚠ **And the structural fact I will state in the report either way:** with a mandated 524,288 B store, the CLU
family can **never** match `knn_proto` on *total* bytes — 524,288 alone already exceeds 475,136. The
"byte-matched" claim on this table rests entirely on the convention that excludes the addressing head.
That convention is the frozen harness's, not mine, and it favours me.

**Still missing from DEV and needed before any claim:** `knn_proto` and the matched-bytes **atom-mean lookup**
(the §C floor) measured **on this same DEV corner**. ⛔ Without them the K-sweep is a curve with no opponents.

## 2026-09-16T20:31Z — ⭐⭐ DEV-03 completes at K=128: **the bottleneck MOVES, and that is the whole finding**

| K | bits | store B | head B | recon MSE | nearest-mode ref | `clu` + NB head | `clu` + disc head | `store_removed` |
|---|---|---|---|---|---|---|---|---|
| 5 (shipped) | 30 | 524,288 | 74,240 | — | 0.4884 (11) | 0.4722 (9) | — | 0.2222 (1) |
| 8 | 48 | 524,288 | 118,784 | 0.0545 | 0.4907 (12) | 0.5116 (11) | 0.5324 (12) | 0.1389 (1) |
| 16 | 96 | 524,288 | 237,568 | 0.0517 | 0.5463 (19) | 0.5625 (16) | 0.5856 (15) | 0.0556 (1) |
| **32** | 192 | 524,288 | 475,136 | 0.0446 | 0.7014 (30) | ⭐ **0.6921 (25)** | 0.6782 (32) | 0.1389 (1) |
| 64 | 384 | 524,288 | 950,272 | 0.0381 | 0.7130 (38) | 0.6713 (38) | 0.6528 (29) | 0.0556 (1) |
| 128 | 768 | 524,288 | 1,900,544 | 0.0289 | ⭐ **0.7963 (42)** | 0.6551 (47) | 0.6528 (44) | 0.0556 (1) |

⭐⭐ **The two columns diverge.** The codebook keeps getting better all the way to K=128 — distortion-optimal
read **0.4884 → 0.7963**, recon MSE **0.0545 → 0.0289** — while the ARM **peaks at K=32 (0.6921) and then
falls** (0.6713, 0.6551). ⇒

> **K ≤ 16: the CODEBOOK is the bottleneck. K ≥ 32: the ADDRESSER is the bottleneck. The crossover is ~K=32.**

⛔ This reverses my 19:40 position *again*, and now with a whole curve rather than one point, so I will state
the honest sequence for the record: at K=5 the addresser had ~0.016 of slack (true, and I over-generalised it);
by K=128 the addresser is leaving **0.14** on the table. **Both statements are measurements; the first one was
only ever true at K=5.**

### ⭐⭐ And the two problems have ONE solution
At K=128 the head has **1,900,544 B** — 4× `knn_proto`'s entire deployed table — and it performs **worse** than
the 475,136 B head at K=32. A head that is simultaneously (a) far too large to declare and (b) losing to a
smaller one is **over-parameterised**. ⇒ ⭐ **A low-rank head is the right answer for BOTH the byte ledger and
the generalisation**, not a byte-saving compromise made against the science. That is now my primary hypothesis
and DEV-05(b) measures it: SVD-truncate the fitted head's `W (57 × K·M)` to rank r and re-score, with the
budget line drawn at **74,240 B — the shipped head, to the byte**.

⚠ Still to measure before any claim: `knn_proto` on this DEV corner (DEV-05a, running). Without it the whole
sweep is a curve with no opponent, and ⛔ I will not name a target number until I have it.

## 2026-09-16T20:33Z — DEV-05: the opponents on the DEV corner, and ⛔ the low-rank LOGIT head is dead

**(a) The opponents, on the same DEV corner, same decoder, same classifier, seed 0:**

| DEV row | aggregate | **distinct tuples / 72** | deployed bytes |
|---|---|---|---|
| `store_removed` | 0.056 – 0.222 | **1** | 524,288 |
| shipped `clu` (K=5, ward, `train_store`) | 0.4722 | 9 | 524,288 + 74,240 head |
| **best constant (realised)** | **0.5278** | **1** | 0 |
| `atom_argmax` — an `x0`-style addressed read, 64 real train rows/well | 0.5833 | 22 | 524,288 + head |
| ⭐ **`atom_mean_lookup` — the §C.1 MATCHED-BYTES LOOKUP FLOOR** | **0.6204** | **28** | 524,288 + head |
| my `clu` K=32, full naive-Bayes head | 0.6921 | 25 | 524,288 + 475,136 head |
| ⭐⭐ **`knn_proto` — the ceiling arm** | **0.7801** | **53** | **475,136** |
| decoder ceiling (real features) | 0.9630 | 72 | — |

⛔⛔ **Read the §C.1 line carefully: the atom-mean lookup on DEV is 0.6204, so my K=32 arm (0.6921) is only
+0.072 above a matched-bytes lookup table** — and it is **0.088 BELOW `knn_proto`**. ⭐ On DEV the ordering
`lookup < mine < knn_proto` is the same ordering the published B table has, which is reassuring for DEV as a
selection instrument: the DEV corner is not flattering me.
⚠ Note also `atom_mean_lookup` (0.6204, 28 tuples) **beats** `atom_argmax` (0.5833, 22) — the *mean* of
addressed atoms beats the *argmax*. ⛔ That is the §C.1 floor being genuinely hard, exactly as the charter
intends.

**(b) ⛔ The low-rank LOGIT head is dead.** SVD-truncating the fitted naive-Bayes `W (57 × K·M)` at K=32:

```
singular values: 387.1 241.0 183.4 176.4 153.3 143.4 139.0 125.4 121.0 117.0 115.1 107.1 ...
spectral mass in the top 8 of 57: 0.7085          ⇒ the head is genuinely HIGH-RANK
rank   2 -> 0.2222 (1/72)    rank   4 -> 0.3056 (12)   rank   7 -> 0.2060 (15)
rank   8 -> 0.1644 (19)      rank  12 -> 0.4954 (19)   rank  16 -> 0.3704 (26)
rank  24 -> 0.6481 (21)      rank  32 -> 0.5949 (18)   rank  48 -> 0.6921 (27)   full -> 0.6921 (25)
```
⛔ Non-monotone and catastrophic below rank ~24, and **no rank that fits the 74,240 B budget is usable**
(rank 7 = 67,132 B scores **0.2060**). ⇒ **Hypothesis rejected: the addressing head cannot be shrunk to the
shipped budget by truncating it in LOGIT space.** Recorded as a dead end, ⛔ not quietly dropped.

## 2026-09-16T20:33Z — ⭐⭐ the design that survives all of the above: **the LAUNCH READ with a K-INDEPENDENT head**

The logit head is `O(K·M)` — that is *why* it grows with K and *why* it over-fits at large K. ⭐ **Put the head
in FEATURE space instead and it becomes `O(D)`, independent of K entirely:**

> `x̂(request) = μ + Σ_f P[offset_f + v_f]`, with `P (57 × 2048)` factorised to rank `r`:
> `head_bytes = (57·r + r·D + D)·4` — ⭐ **the same number at K = 16 and at K = 2048.**
> `codes[k] = argmin_m ‖x̂ restricted to well k − mode_{k,m}‖²`, then the **shipped**
> `gen_store.read_from_codes(store, codes)`.

⭐ Request-only by construction: `x̂` is a function of the request and the head; no feature tensor enters, and
the read's signature is untouched, so `assert_no_bypass` passes unchanged.
⭐⭐ **And it makes the §C.1 test exact, because at FULL rank the head IS `knn_proto`:**
`head_alone = decode(x̂)` is the matched-head lookup floor, at the *same head bytes and no store at all*.
⇒ **`arm > head_alone` is the only way a 524,288 B store earns its bytes on this path.**
⛔ If the store does not beat its own head, it is decorative on this path and I will report exactly that.

⚠ **The honest risk, named in advance:** the nearest-mode reference at K=128 loses **0.167** against the
decoder ceiling *even with oracle codes from the true features* (0.9630 → 0.7963). If quantisation costs that
much on a clean input it may cost as much on `x̂`, in which case `arm < head_alone` everywhere and the answer
is a measured null. ⭐ That is an acceptable outcome and I will not dress it up.
**DEV-06 running:** `K ∈ {16,32,64,128,256} × r ∈ {full, 16, 7}`.

## 2026-09-16T20:38Z — ⛔ DEV-06: on the LAUNCH path the store **does not earn its bytes**, at any K, at any rank

`head_alone` = the rank-r additive request→feature map decoded directly (⭐ at full rank it **is** `knn_proto`,
measured 0.7801 (53/72), 475,136 B — the identity holds exactly, which validates the construction).

| K | nearest-mode ref | r = full (head 475,136 B) | r = 16 (142,912 B) | r = 7 (67,132 B) |
|---|---|---|---|---|
| head alone (no store) | — | **0.7801 (53)** | 0.6065 (35) | 0.5370 (21) |
| 16 | 0.5463 (19) | 0.5810 (16) **Δ −0.199** | 0.5417 (17) **Δ −0.065** | 0.5000 (19) **Δ −0.037** |
| 32 | 0.7014 (30) | 0.6273 (33) **Δ −0.153** | 0.5556 (31) **Δ −0.051** | 0.4931 (13) **Δ −0.044** |
| 64 | 0.7130 (38) | 0.6343 (29) **Δ −0.146** | 0.5764 (25) **Δ −0.030** | 0.5185 (18) **Δ −0.019** |

> ⛔⛔ **Every single cell is negative. Quantising the request's launch point through the 524,288 B store
> makes it WORSE than decoding the launch point directly — at every K and every rank tested.**
> ⭐ And the deficit *shrinks monotonically with K* (−0.037 → −0.044 → −0.019 at r=7), i.e. the store is
> converging on **doing nothing**: at K → D the read is per-channel rounding and Δ → 0. **A store whose best
> case is Δ = 0 is a decoration.** That is the §C.1 verdict for the launch path and I am recording it as one.

`store_removed` collapses to **1 tuple** in every row (0.056–0.139), so the arm is *formally* load-bearing —
⭐ which is exactly why falsifier 3 alone is not enough and why §C.1 exists. ⛔ **A formally load-bearing store
that loses to its own addressing head is still a decoration.**

⭐ **The best arm I have is still the LOGIT head, not the launch:** `K=32` + full naive-Bayes logit head =
**0.6921 (25/72)**, which beats the same store's launch read (0.6273) by +0.065. ⇒ the head that is *trained
to name the code* beats the head that *estimates the feature and then snaps*. Two error sources compound in
the latter.

**Prototype-deviation spectrum** (`P`, 57 × 2048): top-4 66.6 %, top-8 85.0 %, top-16 96.0 %, top-32 99.8 %.
⚠ Yet truncation is expensive in the metric: full 0.7801 → r=16 0.6065 → r=7 0.5370. ⭐ 96 % of the spectral
mass is **not** 96 % of the accuracy.

### Where the remaining loss lives — the probe that decides the next move
Relative quantisation error `MSE / Var(features)`: K=8 **0.155** → K=128 **0.082**. That is a *slow* fall
(the curse of dimensionality: ~400 dims per 6-bit code at K=5, still ~16 dims at K=128), and at K=128 an 8 %
relative error costs **0.167** of accuracy (0.9630 → 0.7963).
⭐ **DEV-07(1), the decisive control:** decode `x + ε` with isotropic `ε` of *exactly the same MSE*.
- If isotropic noise costs about the same ⇒ the codebook is at the information limit ⇒ ⛔ a better codebook
  cannot help and the honest answer is a null.
- If isotropic noise costs far less ⇒ the quantisation error is **structured / decoder-adversarial** ⇒ ⭐ a
  decoder-aware codebook (DEV-04, `diag(JᵀJ)`-weighted) is the lever.
**Queued:** DEV-04 (decoder-sensitivity-weighted codebook at K=32,64) then DEV-07 (noise control, `clu_wide`,
hybrid addresser).

## 2026-09-16T20:53Z — DEV-06 completes at K=256, and DEV-04 returns a **null**

**DEV-06, the last two rungs:**

| K | nearest-mode ref (oracle codes) | launch r=full | launch r=16 | launch r=7 |
|---|---|---|---|---|
| 128 | 0.7963 (42) | 0.6551 (36) Δ −0.125 | 0.5949 (32) Δ −0.012 | 0.5324 (25) Δ −0.005 |
| **256** | ⭐⭐ **0.9074 (56/72)** | 0.6759 (43) Δ −0.104 | 0.5787 (27) Δ −0.028 | 0.5324 (17) Δ −0.005 |

⭐⭐ **The codebook is NOT the limit any more.** At K=256 (8 channels per well) the store reproduces the
features well enough to score **0.9074 with 56 distinct tuples of 72** — *above* `knn_proto` (0.7801) and
close to the decoder ceiling (0.9630) — **at exactly 524,288 B**. ⛔ But that is with **oracle codes read off
the true features**: it is the store's **capacity**, not an arm.
⛔ And every launch cell is still negative, converging to 0 from below as K grows. **On the launch path the
store can only lose or draw. `arm ≤ head_alone` with equality in the limit.** That is not a tuning failure, it
is what quantisation *is*: snapping cannot add information the head did not already have.

**DEV-04 — the decoder-aware codebook: ⛔ a NULL.** `diag(JᵀJ)` of the frozen decoder, Hutchinson, 512 rows
× 8 probes:
```
w_min 0.0153  w_max 0.7274  dynamic range 47.7x  top-10% of coords hold 26.3% of the mass
corr(log w, log Var(features)) = 0.1596       ⇒ the decoder cares about almost a DIFFERENT set of
                                                coordinates than the ones that VARY
K=32 unweighted       mse=0.0446  ref=0.7014(30)  nb=0.6898(24)  disc=0.6806(31)  sr=0.1389
K=32 decoder_weighted mse=0.0462  ref=0.7014(33)  nb=0.6829(38)  disc=0.6875(34)  sr=0.1944
```
⭐ The *premise* checked out — sensitivity and variance are nearly uncorrelated (r = 0.16), so the unweighted
fit really is spending codewords where the decoder is not looking. ⛔ **But weighting the distortion by it
changes nothing** (ref identical to 4 dp; arms move ±0.007, well inside noise on 72 items). ⇒ **Hypothesis
rejected:** a first-order diagonal decoder metric is not the lever. Recorded as a dead end.

### ⚠ The structural tension I can now state with numbers, and it is not good news for the arm
- `head_bytes(logit) = (K·M + Σ_f V_f·K·M)·4 = 232·K·M`, and **`M = 64` is pinned by the 524,288 B store**
  (`store_bytes = M·D·4`). ⇒ `head_bytes = 14,848·K`, unavoidable.
- The store's **capacity curve** (oracle codes): 0.4884 (K=5) · 0.4907 (8) · 0.5463 (16) · 0.7014 (32) ·
  0.7130 (64) · 0.7963 (128) · 0.9074 (256). It crosses `knn_proto`'s 0.7801 at **K ≈ 110**.
- ⇒ **To have the capacity to beat the ceiling arm the store needs K ≳ 110, at which point the logit head
  alone is ≈ 1.6 MB — 3.4× `knn_proto`'s entire deployed table.**
- ⇒ The two things the arm needs — codebook capacity and an affordable head — pull in opposite directions,
  and ⛔ **no parameterisation I have found yet resolves it**: the low-rank logit head fails (DEV-05b), the
  K-independent feature-space head can only lose (DEV-06).

⛔ **I am not calling this yet.** Still to measure, and one of them is byte-legal at the shipped head size:
- ⭐ **`clu_wide`** (the shipped `gen_compose.clu_wide_vector`): the request posterior's weighted **MEAN**
  over modes instead of its argmax. Its reachable set per well is a **64-dimensional subspace**, not 64
  points, so it is not a quantiser and the "snapping cannot add information" argument does not apply to it.
  ⚠ Its channel is **K·M reals, not K integers** — a WIDE channel, declared, ⛔ never quoted in the 30-bit
  column. On DEV the analogous `atom_mean_lookup` (0.6204, 28) already beat `atom_argmax` (0.5833, 22), so
  the mean read is the stronger verb on this data.
- the hybrid addresser (logit + launch distance), and the head's ridge tuned per K.

## 2026-09-16T21:04Z — ⭐⭐⭐ DEV-07(1): **the quantisation error is STRUCTURE, not magnitude.** 0.95 vs 0.70 at the same MSE.

I decoded `x_dev + ε` with `ε ~ N(0, σ²I)` and `σ²` set to **exactly** the quantiser's own MSE at each K, and
put it beside the quantiser at the same MSE:

| MSE | K | ⭐ **isotropic noise of the same energy** | the quantiser (oracle codes) | cost of STRUCTURE |
|---|---|---|---|---|
| 0.0545 | 8 | **0.9213 (61/72)** | 0.4907 (12/72) | **−0.431** |
| 0.0517 | 16 | **0.9005 (64/72)** | 0.5463 (19/72) | **−0.354** |
| 0.0446 | 32 | **0.9491 (64/72)** | 0.7014 (30/72) | **−0.248** |
| 0.0381 | 64 | **0.9375 (69/72)** | 0.7130 (38/72) | **−0.225** |
| 0.0289 | 128 | **0.9329 (65/72)** | 0.7963 (42/72) | **−0.137** |

*(decoder ceiling on clean features: 0.9630, 72/72)*

> ⭐⭐⭐ **Perturbing the features by the SAME amount of energy costs almost nothing (0.963 → ~0.93) if the
> perturbation is isotropic, and costs up to 0.43 if it is the codebook's own residual.**
> ⇒ ⛔ **The codebook is NOT at an information limit and I was wrong to fear it was.** Its error is
> **low-dimensional and aligned with the directions the decoder actually reads.** k-means residuals live in
> the leading within-cluster principal directions; isotropic noise spreads the same energy over 2048
> directions so almost none of it lands in the handful the decoder uses.

⭐ **And this explains the DEV-04 null exactly.** `diag(JᵀJ)` is a **coordinate-wise** weight. The harmful
directions are *not* coordinate-aligned, so a diagonal metric cannot see them — which is why re-weighting
changed nothing while the premise (corr(log w, log var) = 0.16) was correct.
⇒ ⭐⭐ **Next lever, and now a principled one: quantise under the FULL per-well metric `A_k = (JᵀJ)|_{well k}`,
not its diagonal.** The Hutchinson probes `g = Jᵀu ~ N(0, JᵀJ)` already give me samples; their per-well
covariance *is* `A_k`. The Lloyd update stays the plain mean under any fixed positive-definite metric
(`∂/∂m Σ (v−m)ᵀA(v−m) = 0 ⇒ m = mean`), so only the ASSIGNMENT changes ⇒ ⭐ **zero deployed bytes**, and the
metric is never stored (the codes come from the request at read time, not from features).

**DEV-07(2,3), same run:** `clu_wide` at K=32 = **0.6574 (22/72)** — ⛔ *worse* than the argmax read (0.6921,
25/72), so the posterior-mean verb does **not** rescue the request-driven read at K=32. The hybrid addresser
(`log P(m|req) − λ‖x̂_k − m‖²`) peaks at λ=0.3 with **0.6991 (25/72)** vs 0.6921 at λ=0 — **+0.007, inside
noise on 72 items**. ⛔ Neither is a lever.

## 2026-09-16T21:14Z — DEV state of play (seed 0, one corner) — ⭐ every number carries its distinct-tuple count

**References on the DEV corner**

| reference | aggregate | distinct tuples / 72 | bytes |
|---|---|---|---|
| `store_removed` (laundering control) | 0.056 – 0.222 | **1** | 524,288 |
| best constant output (structural bound **and** realised) | **0.5278** | **1** | 0 |
| `atom_mean_lookup` — §C.1 matched-bytes lookup | 0.6204 | 28 | 524,288 + head |
| ⭐ **`knn_proto` — the ceiling arm** | **0.7801** | **53** | **475,136** |
| decoder ceiling (real features) | 0.9630 | 72 | — |

**Everything I have tried, best value per family**

| family | best DEV aggregate | tuples | head B | verdict |
|---|---|---|---|---|
| shipped `clu`, K=5, ward, `train_store` | 0.4722 | 9 | 74,240 | ⛔ below the constant floor |
| K-sweep, naive-Bayes logit head (**K=32**) | ⭐ **0.6921** | 25 | 475,136 | ⭐ best arm so far; ⛔ head 6.4× shipped |
| K-sweep, discriminative logit head | 0.6875 (K=32, w) / 0.5856 (K=16) | 34 / 15 | same | helps at K≤16, hurts at K≥32 |
| hybrid addresser (logit + launch), K=128 λ=3 | 0.6991 | 36 | 1,900,544 | +0.007 — inside noise |
| launch read (K-independent feature head) | ≤ `head_alone` everywhere | — | 67 k – 475 k | ⛔ store never earns its bytes |
| low-rank LOGIT head (SVD truncation) | 0.2060 at the 74,240 B budget | 15 | 67,132 | ⛔ dead |
| `clu_wide` (posterior-mean over modes), K=32 / K=128 | 0.6574 / 0.6412 | 22 / 31 | 475 k / 1.9 M | ⛔ worse than argmax |
| decoder-**diagonal**-weighted codebook | 0.6829 (K=32) | 38 | — | ⛔ null |

**Codebook CAPACITY (oracle codes from the true features — ⛔ diagnostics, never arms)**
0.4884 (K=5) · 0.4907 (8) · 0.5463 (16) · 0.7014 (32) · 0.7130 (64) · 0.7963 (128) · ⭐ **0.9074 (256, 56/72)**.
**Per-well code agreement of the request-driven head vs those oracle codes:** 0.903 (K=8) · 0.847 (16) · 0.830 (32).

**Where that leaves me, stated without varnish at the 2-hour mark:**
- ⭐ The store is **not** capacity-limited: at 524,288 B and K=256 it reproduces the corner at 0.9074 (56/72).
- ⛔ **No request-only addresser I have built gets past ~0.70**, and `knn_proto` is at 0.7801.
- ⛔ The two things that would fix it pull against each other: capacity wants K ≳ 110, and the addressing head
  costs `14,848·K` B.
- ⭐ **But DEV-07 says the loss is structure, not information** — isotropic noise of the same energy costs 0.03
  where the codebook's own residual costs up to 0.43. So there is a large, *identified* inefficiency left.
**Running:** DEV-08 (the shipped K=5 cell read as a posterior-mean, at the exact shipped byte ledger),
DEV-09 (**full** per-well decoder metric `A_k = (JᵀJ)|_k`, zero deployed bytes), DEV-10 (⭐ the shrinkage
repair: `mode ← mean_k + α(mode − mean_k)` — a centroid is an average, so the read is systematically
low-contrast, which is exactly the coherent low-dimensional error DEV-07 says is expensive; zero bytes).

## 2026-09-16T21:17Z — ⭐⭐⭐ DEV-08: **THE READ VERB IS THE DEFECT.** The modes already span the answer.

The SHIPPED cell, untouched — `K=5`, `M=64`, ward grouping, the shipped `train_store`,
**store 524,288 B · addressing head 74,240 B · total 598,528 B · 30.0 bits/read** (ledger printed by the
shipped `deployed_byte_ledger`) — read five ways plus three references:

| DEV row (shipped store, shipped bytes) | aggregate | **distinct tuples / 72** |
|---|---|---|
| `store_removed` (argmax and wide) | 0.2222 | **1** |
| ⛔ **`argmax` — the shipped read (30 bits)** | **0.4722** | **9** |
| `wide` — posterior MEAN over modes, naive-Bayes head | 0.5093 | 13 |
| `wide_disc` — posterior mean, discriminative head | 0.5255 | 21 |
| `argmax_disc` — argmax, discriminative head (same 74,240 B) | **0.5301** | 17 |
| *best constant* | *0.5278* | *1* |
| ⛔ `ref_argmax` — nearest mode to the TRUE feature (diagnostic) | 0.4884 | 11 |
| ⛔ `ref_simplex` — best CONVEX combination of the modes (diagnostic) | 0.5509 | 12 |
| ⭐⭐⭐ ⛔ **`ref_span` — best LINEAR combination of the modes (diagnostic)** | **0.8264** | **42** |
| `knn_proto` (475,136 B) | 0.7801 | 53 |
| decoder ceiling | 0.9630 | 72 |

> ⭐⭐⭐ **The 64 modes of each well span a subspace that carries the corner at 0.8264 — ABOVE `knn_proto` —
> and the shipped store already contains them.** Vertices give 0.4884, the convex hull gives 0.5509, the
> **span** gives 0.8264. ⛔ **The programme's 0.5949 is not a codebook failure and not an addressing failure.
> It is the read being restricted to ONE VERTEX of a 64-dimensional subspace it already owns.**

⭐ And the shipped code already has the verb: `gen_compose.clu_wide_vector(store, weights)` computes
`x_k = Σ_m w_{k,m}·mode_{k,m}` as a **plain matmul with no simplex constraint** and takes no feature
argument. ⛔ No new read function, no new bypass surface.

### ⇒ DEV-11, launched: the store as a DICTIONARY
Head = an **additive** coefficient table of the shipped head's exact shape `(1 + Σ_f V_f) × K × M`
— **74,240 B at K=5, M=64: the shipped addressing head to the byte.**
⭐ For a block-diagonal basis the least-squares fit of that head is *identical* to projecting the per-factor
prototype table onto each well's mode span, so the arm is exactly **`knn_proto` compressed into the store's
own basis**, and the store is doing the compressing.

| | formula | at K=5, M=64 |
|---|---|---|
| store | `M·D·4` | 524,288 B |
| head | `(1 + Σ_f V_f)·K·M·4` | 74,240 B |
| **total** | `4M(D + 58K)` | 598,528 B |

⭐⭐ **`total < knn_proto`'s 475,136 B whenever `M ≤ 50` at K=5** (M=32 ⇒ **299,264 B**, 0.63× `knn_proto`).
So for the first time there is a family of configurations that is **strictly under the ceiling arm's byte
ledger**, not over it. ⇒ ⭐ a WIN here would be a win at **fewer** bytes, not more.

**Every row of DEV-11 carries its own §C controls, by construction:**
- §C.1 floor: **the same linear read with 64 REAL TRAIN ROWS as the basis** instead of learned modes, same
  bytes. ⇒ "does *learning* the dictionary beat *storing rows*".
- §C.2 control: the same coefficients with every mode replaced by its well mean.
⚠ **Declared up front:** this read's channel is `K·M` reals, ⛔ **not 30 bits** — it is a WIDE channel and I
will never quote it in the 30-bit column. At K=5, M=64 it is **320 reals against `knn_proto`'s 2048**, i.e.
6.4× narrower than the ceiling arm's, and the shipped harness already carries such an arm (`clu_wide`, R27).

## 2026-09-16T21:26Z — ⭐⭐ DEV-11: the dictionary read works, beats the §C.1 atom floor — and is **STRUCTURALLY CAPPED at `knn_proto`**

`arm` = the additive coefficient head over the store's own basis (⭐ = the prototype table projected onto each
well's mode span). `cap` = the same span applied to the TRUE feature (⛔ diagnostic). `atomfloor` = ⭐ **the
§C.1 control: the identical read with M REAL TRAIN ROWS as the basis instead of learned modes.**
`sr` = §C.2 laundering control. `knn_proto` (ridge form, same rows) = **0.7755 (52/72) at 475,136 B.**

| K | M | **arm** | tuples | cap (diag.) | ⭐ atom-basis floor | `sr` | store B | head B | **total B** | vs knn bytes |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 16 | 0.5278 | 31 | 0.5417 | **0.6667** | 0.0764 | 131,072 | 29,696 | 160,768 | 0.34× |
| 8 | 32 | 0.6227 | 30 | 0.6667 | **0.6806** | 0.1389 | 262,144 | 59,392 | 321,536 | 0.68× |
| 8 | 47 | 0.6667 | 39 | 0.8079 | **0.6898** | 0.1389 | 385,024 | 87,232 | 472,256 | 0.99× |
| 16 | 16 | 0.5949 | 34 | 0.6389 | **0.6690** | 0.2616 | 131,072 | 59,392 | 190,464 | 0.40× |
| 16 | 32 | ⭐ 0.7384 | 38 | 0.8542 | 0.6875 | 0.2986 | 262,144 | 118,784 | 380,928 | 0.80× |
| 16 | 39 | ⭐ 0.7431 | 44 | 0.8565 | 0.7269 | 0.1968 | 319,488 | 144,768 | 464,256 | 0.98× |
| 32 | 16 | 0.6968 | 37 | 0.7917 | 0.6806 | 0.3611 | 131,072 | 118,784 | 249,856 | 0.53× |
| ⭐ **32** | **30** | ⭐⭐ **0.7662** | **45** | 0.9468 | 0.7384 | 0.3171 | 245,760 | 222,720 | **468,480** | **0.99×** |
| 8 | 64 | 0.6944 | 45 | 0.8310 | 0.7106 | 0.1898 | 524,288 | 118,784 | 643,072 | 1.35× |
| 16 | 64 | 0.7755 | 49 | 0.9514 | 0.7546 | 0.0648 | 524,288 | 237,568 | 761,856 | 1.60× |
| 32 | 64 | 0.7755 | 52 | **0.9630** | 0.7755 | 0.3241 | 524,288 | 475,136 | 999,424 | 2.10× |

**What is now established, and it cuts both ways.**
- ⭐ **The dictionary read beats the §C.1 atom-basis floor** once the partition is fine enough: K=16 M=32
  (+0.051), K=16 M=39 (+0.016), K=32 M=16 (+0.016), **K=32 M=30 (+0.028)**. ⇒ *learning* the dictionary beats
  *storing rows* at the same bytes. ⛔ At K=8 it does not (the atom basis wins at every M) — recorded.
- ⭐ Best configuration **strictly under `knn_proto`'s byte ledger**: **K=32, M=30 → 0.7662 (45/72) at
  468,480 B = 0.99× `knn_proto`'s 475,136 B**, against `knn_proto`'s 0.7755 (52/72). **Δ = −0.0093.**
- ⛔⛔ **And the last row is the proof that this can never be a win.** At K=32, M=64 the arm, the atom floor
  **and** `knn_proto` all read **0.7755** — identical to four decimals. At that capacity the per-well spans
  cover the feature space, the projection becomes the identity, and **the arm IS `knn_proto`**.
  ⇒ **the linear dictionary read converges to `knn_proto` FROM BELOW and is structurally capped by it.**
  That is not a tuning limit; it is what the construction is: the arm is `knn_proto`'s own table projected
  onto a subspace, and a projection cannot add.

### ⭐⭐⭐ But the same fact is the opening, and it is the first idea in this run that is not capped
`knn_proto` is the best **ADDITIVE** request→feature model. Shapes3D plainly has **interactions** — a cube at
30° and a cylinder at 30° are not the same picture — and an additive model cannot express one.
- A pairwise-interaction head in **raw feature space** needs `1380 × 2048 × 4` = **11.3 MB**. ⛔ Impossible at
  this budget. That is why nobody on this table has one.
- The same head in the **store's basis** needs `1380 × K·M × 4`. At K=16, M=16 that is **1.4 MB**; restricted
  to the pairs that matter (`shape×orientation` 60 cells + `shape×scale` 32 = 92 columns) it is
  `92 × 256 × 4` = **94,208 B**, on top of a 59,392 B additive head and a 131,072 B store —
  **284,672 B total, 0.60× `knn_proto`.**
⇒ ⭐⭐ **The store's bytes buy something `knn_proto` structurally cannot afford: a non-additive request model.**
⛔ Conditional on DEV-12 showing the interactions are real and estimable from the pool. That is running now,
in raw feature space, as the unaffordable-but-decisive reference.

## 2026-09-16T21:26Z — ⭐⭐⭐⭐ DEV-12: **`knn_proto` IS BEATABLE, and by a mile — the ceiling arm is ADDITIVE and the corner is not**

A request→feature model fitted on the store pool, decoded through the frozen decoder, read by the frozen
classifier. No store anywhere in this experiment — it is a pure statement about the **hypothesis class**.

| request→feature model | columns | aggregate | **distinct tuples / 72** | bytes if deployed |
|---|---|---|---|---|
| **additive** (= `knn_proto`) | 58 | 0.7755 | 52 | 475,136 |
| k-NN in factor space, k=1 / 5 / 20 | — | 0.6319 / 0.6991 / 0.7292 | 26 / 58 / 47 | — |
| + **pairwise**, ridge 1 | 1380 | 0.9815 | 69 | 11,304,960 |
| + pairwise, ridge 10 | 1380 | 0.9838 | 70 | 11,304,960 |
| ⭐⭐ **+ pairwise, ridge 100** | 1380 | ⭐⭐ **0.9884** | ⭐⭐ **72 / 72** | ⛔ **11,304,960** |
| + pairwise, ridge 1000 | 1380 | 0.9028 | 65 | 11,304,960 |
| ⛔ *oracle additive refitted ON the corner (a cheat)* | 58 | 0.9699 | 72 | — |
| *decoder ceiling on the true features* | — | *0.9630* | *72* | — |

> ⭐⭐⭐ **A pairwise-interaction request model reaches 0.9884 with 72/72 distinct tuples on a corner it has
> never seen — 0.21 above `knn_proto` and above the decoder's own ceiling on the true features.**

**Three things this settles.**
1. ⛔⛔ **`knn_proto` is not near any ceiling. It is simply ADDITIVE**, and Shapes3D is not: a cube at 30° and
   a cylinder at 30° are different pictures, and `μ + Σ_f (μ_{f,v} − μ)` cannot say so. The `oracle_additive`
   cheat row proves the point from the other side: even **fitted on the corner itself** the additive class
   only reaches 0.9699 — ⭐ so the gap `0.7755 → 0.9884` is **model class**, not estimation.
2. ⭐ It is honest generalisation, ⛔ not leakage. Montero's `recomb2element` holds out **one six-way
   conjunction**; every 2-way marginal it needs is observable elsewhere in train. A pairwise model
   generalising into the held-out cell is exactly the capability the split is designed to test.
3. ⛔ **But 1,380 columns × 2048 features × 4 B = 11.3 MB — 24× `knn_proto`'s entire table.** In raw feature
   space a non-additive request model is unaffordable at this budget. **That is why nobody on this table has
   one, and it is the reason the table looks the way it does.**

### ⭐⭐⭐ THE THESIS, now fully formed — and it is the first one in this run that is not structurally capped
> **The store's bytes buy a hypothesis class that raw feature space cannot afford.**
> A pairwise head costs `n_cols × D × 4` in feature space and `n_cols × K·M × 4` in the store's basis.
> The store compresses `D = 2048` to `K·M`, so it buys the **interaction terms** — and the interaction terms
> are worth **+0.21**. `knn_proto` cannot do this: its "store" *is* its head, so widening its class widens its
> ledger 24×.

**Budget arithmetic for the arm, written before I measure it.** With `n_cols` request columns,
`total = M·D·4 + n_cols·K·M·4 = 4M(2048 + n_cols·K)`; `knn_proto` = 475,136 B ⇒
| interaction set | `n_cols` | byte-legal configurations (total < 475,136) |
|---|---|---|
| all 15 pairs | 1380 | K=4 M≤15 · K=8 M≤9 · K=16 M≤4 — ⚠ very small bases |
| `shape×orientation` + `shape×scale` only (+ additive) | 150 | K=8 M≤36 · K=16 M≤26 · **K=32 M≤17** |

**DEV-13, next:** rank the 15 pairs in raw feature space (which interactions actually carry the +0.21), then
build the arm in the store's basis at a byte-legal `(K, M, pair-set)` with its §C.1 atom-basis floor and its
§C.2 laundering control on every row.

## 2026-09-16T21:36Z — DEV-13: **which interactions carry the +0.21**, and ⛔ why a FULL interaction head in the store's basis fails

**Part A — additive + ONE pair, in raw feature space (⛔ reference, not an arm). Additive alone = 0.7755 (52/72).**

| added pair | aggregate | tuples | | added pair | aggregate | tuples |
|---|---|---|---|---|---|---|
| ⭐ **floor_hue × scale** | **0.8773** | 64 | | object_hue × shape | 0.8032 | 54 |
| ⭐ **object_hue × scale** | **0.8565** | 57 | | wall_hue × orientation | 0.7847 | 51 |
| ⭐ **shape × orientation** | **0.8171** | 48 | | floor_hue × object_hue | 0.7824 | 50 |
| floor_hue × shape | 0.8102 | 48 | | object_hue × orientation | 0.7778 | 51 |
| wall_hue × shape | 0.8032 | 59 | | floor_hue × orientation | 0.7755 | 56 |
| scale × orientation | 0.8032 | 50 | | floor_hue × wall_hue | 0.7708 | 50 |
| | | | | wall_hue × scale · scale × shape | 0.7662 · 0.7639 | 49 · 50 |

**Cumulative (top-j pairs, raw feature space, ⛔ bytes are prohibitive):**
top1 **0.8773** (64/72, 1.13 MB) · top2 **0.9444** (68/72, 1.79 MB) · top3 **0.9514** (70/72, 2.28 MB) ·
top5 0.9583 (69) · top6 **0.9792** (69, 3.26 MB) · top8 0.9792 (71) · top10 0.9630 (69).
⭐ **Two pairs are worth +0.169 over `knn_proto`; six are worth +0.204.** The interactions that matter are
**hue × scale** (an object's apparent size and its colour are coupled through the rendering) and
**shape × orientation** — physically sensible, not an artefact.

**Part B — ⛔ the full interaction head in the store's basis STARVES the basis, and it fails.**
`total = 4M(D + n_cols·K)` means a 138- or 278-column head eats the budget the store needs:

| config | arm | tuples | §C.1 atom floor | `sr` | store B | head B | total B |
|---|---|---|---|---|---|---|---|
| K=32 M=18, top1 (138 cols) | 0.7616 | 44 | **0.7662** ⛔ floor wins | 0.2824 | 147,456 | 317,952 | 465,408 |
| K=16 M=27, top1 | 0.7523 | 47 | 0.7037 | 0.0579 | 221,184 | 238,464 | 459,648 |
| K=64 M=10, top1 | 0.7083 | 42 | **0.7593** ⛔ floor wins | 0.3912 | 81,920 | 353,280 | 435,200 |
| K=16 M=21, top2 (218 cols) | 0.6829 | 35 | 0.6505 | 0.1435 | 172,032 | 292,992 | 465,024 |
| K=32 M=10, top3 (278 cols) | 0.7176 | 34 | 0.6806 | 0.2731 | 81,920 | 355,840 | 437,760 |
| K=8 M=25, top4 (318 cols) | 0.6690 | 42 | **0.8310** ⛔ floor wins | 0.0926 | 204,800 | 254,400 | 459,200 |

⛔ **Raw top1 0.8773 → 0.7616 once projected into a 576-dimension store basis.** The head is paying
`n_cols × K·M` and there is nothing left for `M`. ⇒ **the full interaction head is the wrong
parameterisation**, and I am recording the failure rather than quietly re-scoping.

⭐ **The repair, and it is the whole point of having a store:** pay `r·K·M` for the interactions instead of
`n_cols·K·M` — a rank-`r` factorisation of the interaction block *in the store's coefficient space*:
`head = (58·K·M + n_pair_cols·r + r·K·M)·4`. DEV-14 is running it, with **four** substitute bases at
identical bytes on every row: learned **modes** · **atoms** (real train rows, §C.1) · **PCA** (⛔ the sharpest
trivial substitute — not charter-legal as a store, but exactly the laundering control) · **random** Gaussian
directions (the D-matched null) · plus **store_removed**.

⚠ **A noise floor I must respect from here on.** DEV is **72 items × 6 factors**. A single factor accuracy at
p ≈ 0.8 has SE ≈ 0.047; the six-factor mean has SE ≈ 0.02 if the factors were independent, so ⛔ **DEV
differences below ~0.04–0.05 are not reliable on one corner and one seed.** Before any scored run, the chosen
configuration is re-measured on **several DEV corners and all three seeds**.

## 2026-09-16T21:49Z — ⭐⭐⭐ DEV-14: a **rank-r head over the store's basis** clears `knn_proto` by +0.13 **under its byte ledger**

Capacity first — the pairwise model projected into the store's basis (⛔ head bytes ignored, a diagnostic):

| | raw | K=16 M=32 (512) | K=32 M=26 (832) | K=32 M=30 (960) | K=16 M=48 (768) | K=64 M=16 (1024) | K=32 M=48 (1536) |
|---|---|---|---|---|---|---|---|
| top-2 pairs | 0.9444 (68) | 0.8287 (48) | 0.8681 (54) | 0.8889 (57) | 0.8750 (62) | 0.8704 (52) | **0.9375 (65)** |
| top-3 pairs | 0.9514 (70) | — | 0.8935 (59) | 0.9120 (59) | 0.9097 (66) | — | — |

⇒ ⭐ a ~1,000-dimension store basis keeps most of the interaction model's gain. The problem was never the
basis; it was paying `n_cols × K·M` for the head.

**With the interaction block factorised to rank r — the byte-legal rows (`total < knn_proto`'s 475,136 B),
best 25 of the sweep, every basis shown because the basis IS the control:**

| K | M | r | basis | pairs | ridge | aggregate | tuples | total B |
|---|---|---|---|---|---|---|---|---|
| ⭐ **32** | **26** | **16** | **modes** | 2 | 10 | ⭐⭐ **0.9097** | **61** | **469,504** |
| 64 | 16 | 8 | ⚠ **atoms** | 2 | 10 | **0.9051** | 54 | 406,528 |
| 32 | 26 | 16 | modes | 3 | 10 | 0.9051 | 63 | 473,344 |
| 64 | 16 | 16 | ⚠ atoms | 2 | 10 | 0.9005 | 58 | 444,416 |
| 64 | 16 | 16 | modes | 2 | 10 | 0.8889 | 56 | 444,416 |
| 64 | 16 | 8 | modes | 2 | 10 | 0.8727 | 54 | 406,528 |
| 64 | 16 | 16 | pca | 3 | 10 | 0.8634 | 53 | 448,256 |
| 16 | 32 | 16 | modes | 2 | 10 | 0.8565 | 52 | 423,936 |

*(`knn_proto` on this corner: **0.7755–0.7801, 52–53/72, 475,136 B**. Best constant 0.5278, 1 tuple.
Decoder ceiling 0.9630, 72 tuples.)*

⭐ **0.9097 with 61 distinct tuples of 72, at 469,504 B = 0.988× `knn_proto`'s ledger. Δ = +0.134 over the
ceiling arm, against a `m = 0.020` win margin.**

### ⛔⛔ TWO REASONS I AM NOT CALLING THIS A RESULT
1. ⛔ **The `atoms` basis is right behind it (0.9051), and at `K=64, M=16, r=8` the atom basis BEATS the
   learned modes (0.9051 vs 0.8727).** ⭐ §C.1 requires beating the matched-bytes lookup **at the same
   configuration, beyond seed spread**. On the evidence so far the learned dictionary's advantage over
   *storing real rows* is **+0.005 at best and negative at some configurations** — ⛔ that is not a §C.1 pass,
   and if it holds up the honest verdict is *"the gain is the FACTORISED READ, not the learned store"*.
2. ⛔⛔ **I have just taken a maximum over several hundred configurations on 72 items.** The DEV noise floor is
   ~0.04–0.05 (SE ≈ 0.02 per six-factor mean at n=72). **A maximum over that many cells is guaranteed to be
   optimistically biased**, and reporting 0.9097 as *the* number would be exactly the "manufactured win" the
   charter calls the one unrecoverable failure.

⇒ ⭐ **Next, and before any scored run: re-measure a SHORT LIST on several DEV corners and all three seeds,
with the basis controls at matched configuration.** The number that goes into a PREREG is the one that
survives that, ⛔ not the maximum of the search.

## 2026-09-16T21:57Z — ⛔ DEV-15 rejected: the **additive block must stay at full rank**

DEV-15 factorised the WHOLE coefficient matrix (additive + interactions) to rank r. It collapses:

```
[p2 ridge10 K32 M48] modes=0.2454  atoms=0.3472  pca=0.5116  random=0.6111  mean=0.2569
[p2 ridge10 K32 M30] modes=0.6667  atoms=0.5880  pca=0.8102  random=0.7338  mean=0.3264
[p2 ridge10 K16 M48] modes=0.4907  atoms=0.4977  pca=0.6181  random=0.6273  mean=0.0671
```
⛔ Far below DEV-14's 0.9097 at comparable sizes — and it is **the same failure as DEV-05b**: truncating the
ADDITIVE part of a request head destroys it, because the additive part is genuinely high-rank (it is
`knn_proto`, and `knn_proto` is not low-rank). ⭐ Only the **interaction** block factorises cleanly.
⇒ the parameterisation is fixed, and it is DEV-14's:

> `head_bytes = (n_additive_cols · K·M  +  n_pair_cols · r  +  r · K·M) · 4` · `store = M·D·4`
> — additive block full rank, interaction block rank `r`.

⚠ Also visible in those rows, and I am not hiding it: on the **wrong** parameterisation the `pca` and
`random` bases beat the learned `modes`. ⛔ That is a warning that the basis controls are live and that the
learned store is not automatically the best dictionary.

**DEV-16 launched:** six fixed configurations × **six Montero-shaped DEV corners**, five bases at matched
configuration on every row, paired against `knn_proto` corner by corner. ⛔ Nothing is re-selected inside it:
the configurations, the pair ranking and the ridge are all inherited. It reports mean ± sd across corners and
a paired win count, ⛔ not a maximum.

## 2026-09-16T22:02Z — ⭐ removing the last selection risk: **ALL 15 pairs, no pair selection at all**

DEV-13 ranked the factor pairs on ONE corner and DEV-14/16 used the top 2–3. ⛔ That is a selection made on a
72-item DEV corner, and it may not transfer to B. ⭐ The head does not need it: with the interaction block
factorised to rank `r`, **all 15 pairs fit under `knn_proto`'s ledger**, so the pair set can simply be
*everything* and there is nothing left to tune to a corner.

`n_additive = 58` · `n_pair(all 15) = 1322` · `head = (58·K·M + 1322·r + r·K·M)·4` · `store = M·D·4`:

| K | M | r | head B | store B | **total B** | vs `knn_proto` 475,136 B |
|---|---|---|---|---|---|---|
| 32 | 26 | 8 | 261,952 | 212,992 | **474,944** | ⭐ 0.9996× |
| 32 | 26 | 4 | 227,488 | 212,992 | **440,480** | ⭐ 0.927× |
| 32 | 24 | 12 | 278,496 | 196,608 | **475,104** | ⭐ 0.99993× |
| 64 | 14 | 8 | 278,848 | 114,688 | **393,536** | ⭐ 0.828× |
| 16 | 30 | 8 | 169,024 | 245,760 | **414,784** | ⭐ 0.873× |

⭐ **DEV-17 runs exactly these six, on the same six corners, with the same five bases.** The only quantities
selected on DEV are then `(K, M, r, ridge)` — four numbers — and ⛔ **no pair set, no factor ranking, and
nothing per-corner**.

**Also written: `chlu/core/gen_factored_read.py` on my branch** — the production form of the arm.
`FactoredRequestHead.coefficients(requests)` (⛔ signature takes requests, nothing else) feeding the **shipped**
`gen_compose.clu_wide_vector(store, weights)`, plus `assert_request_only_factored`, which mirrors the shipped
`assert_request_only` at the same strength: signature check · independent re-implementation of the read,
bit-compared · coefficient provenance re-derived · bit-identity under feature replacement. Its
`byte_ledger()` and `channel_ledger()` report the head's bytes and the **`K·M`-real channel width** so the
wide channel can never be quoted in the 30-bit column.
⛔ `chlu/experiments/exp_gen_cell.py` and every other frozen-harness file remain untouched.

## 2026-09-16T22:12Z — ⛔⛔ DEV-16: **the +0.13 did NOT survive six corners.** It was +0.0147.

Six fixed configurations × six Montero-shaped DEV corners, five bases at matched configuration, paired
against `knn_proto` corner by corner. `knn_proto` across the six corners: **0.8302 ± 0.0770**
(per corner 0.7801 · 0.9190 · 0.8750 · 0.8287 · 0.8889 · 0.6898) — ⭐ reassuringly close to its published
B value 0.8650 ± 0.0122, so the DEV corners are a fair instrument. Best constant 0.5278 on every corner.

| config (2–3 pairs) | ⭐ modes | tuples | atoms (§C.1) | pca | random | mean (§C.2) | **Δ vs knn** | paired wins |
|---|---|---|---|---|---|---|---|---|
| K32 M26 r16 p2 | **0.8449 ± 0.0778** | 56.5 | 0.8079 | 0.7276 | 0.6767 | 0.2461 | **+0.0147** | **3/6** |
| K32 M26 r8 p2 | 0.8418 ± 0.0816 | 55.8 | 0.8036 | 0.7261 | 0.6782 | 0.2485 | +0.0116 | 2/6 |
| K64 M16 r16 p2 | 0.8353 ± 0.0836 | 56.7 | 0.7948 | 0.7801 | 0.6910 | 0.2847 | +0.0050 | 3/6 |
| K64 M16 r8 p2 | 0.8245 ± 0.1077 | 55.2 | 0.7936 | 0.7693 | 0.6740 | 0.2755 | −0.0058 | 3/6 |
| K32 M26 r16 p3 | 0.8198 ± 0.0579 | 56.7 | 0.7924 | 0.7261 | 0.6690 | 0.2450 | −0.0104 | 2/6 |
| K16 M32 r16 p2 | 0.7897 ± 0.0749 | 48.0 | 0.7365 | 0.6370 | 0.6169 | 0.2006 | −0.0405 | 1/6 |

> ⛔⛔ **Corner 0 gave +0.134 (0.9097 vs 0.7801). Six corners give +0.0147 with 3/6 paired wins.**
> ⭐ **That is exactly the maximum-over-a-search bias I named at 21:49, arriving as predicted** — which is why
> the search maximum was never going to be the reported number. Recorded in full.

**What DOES survive the six corners, and it is not nothing:**
- ⭐ **§C.1 passes at every configuration**: the learned `modes` dictionary beats the matched-bytes `atoms`
  dictionary by **+0.027 to +0.053** (3–5 of 6 paired). ⇒ *learning* the dictionary beats *storing rows*.
- ⭐ **§C.2 passes hard**: replacing every mode by its well mean collapses the arm to **0.20–0.28**, far below
  the 0.5278 constant floor.
- ⭐ **The `random` basis (the D-matched null) sits at 0.62–0.69**, well below `modes`. The partition and the
  fitted dictionary are both doing real work.
- ⭐ The `pca` basis (0.73–0.78) is **below** `modes` on the cross-corner mean at every configuration, which
  answers the sharpest laundering question: ⛔ the store is not just "any 800-dimensional subspace".
- ⭐ Distinct tuples **55–57 of 72** against `knn_proto`'s 48–57, at ≤ 0.99× its bytes.

**But Δ = +0.0147 with 3/6 paired wins is a WASH, not a win**, and against the ceiling rule (`m = 0.020`;
TIE needs `0 < Δ ≤ m` with **3/3 same sign**) it is not even a favourable tie — the sign is not consistent.

### ⚠ The one measurement that decides whether anything is left here
DEV-12's `+0.21` for the pairwise model was measured on **corner 0 only**. If the raw pairwise model is also
only `+0.02` across six corners, then ⛔ the interaction advantage was corner-specific too and the honest
answer for this hub is a null. If it is still `+0.10` or more, then the gap is my **projection** losing it and
basis capacity is the lever. ⭐ DEV-18 measures exactly that, per corner, in raw feature space.

## 2026-09-16T22:30Z — ⛔ DEV-17 (all 15 one-hot pairs) is worse; ⭐⭐⭐ DEV-18 says the interaction advantage is REAL, 6/6

**DEV-17 — all 15 pairs, no pair selection, six byte-legal configurations, six corners:**
every configuration lands at **0.713–0.783** with `Δ vs knn = −0.05 … −0.12` and **0–1 of 6** paired wins.
⛔ Rejected: with 1,322 one-hot pair columns the rank-4…12 factorisation throws the interactions away
faster than it buys them. ⭐ §C.1 still passes everywhere (modes − atoms = **+0.021 … +0.065**, up to 6/6).

**⛔ And the cross-validated verdict on DEV-16's best, which is the number that matters.**
Selection corners `c0–c2` (where the pair ranking was made) vs **held-out corners `c3–c5`**:

| config | SEL modes | SEL knn | **HELD-OUT modes** | HELD-OUT atoms | HELD-OUT pca | HELD-OUT knn | **Δ** | wins |
|---|---|---|---|---|---|---|---|---|
| K32 M26 r16 p2 | 0.8927 | 0.8580 | **0.7971** (55.3 tuples) | 0.7485 | 0.7616 | 0.8025 | **−0.0054** | 1/3 |
| K64 M16 r16 p2 | 0.8789 | 0.8580 | 0.7917 (58.7) | 0.7299 | 0.7747 | 0.8025 | −0.0108 | 1/3 |
| K32 M26 r8 p2 | 0.8927 | 0.8580 | 0.7909 (53.7) | 0.7323 | 0.7454 | 0.8025 | −0.0116 | 0/3 |
| K16 M32 r16 p2 | 0.8364 | 0.8580 | 0.7431 (44.0) | 0.7222 | 0.6211 | 0.8025 | −0.0594 | 0/3 |

⇒ ⛔ **On corners it was not selected on, the arm does NOT beat `knn_proto` — it is −0.005, a tie at best.**
⭐ §C.1 survives the split cleanly: **modes − atoms = +0.049** on the held-out corners, and modes − pca = +0.036.

## ⭐⭐⭐ DEV-18 — the interaction advantage is **not** corner-specific. 6 corners, 6/6.

Raw request→feature models, no store anywhere, per corner, ridge tuned per corner:

| corner | additive (= `knn_proto`'s class) | tuples | **+ all 15 pairwise** | tuples |
|---|---|---|---|---|
| c0 | 0.7824 | 50 | **0.9884** | 72 |
| c1 | 0.9421 | 55 | **1.0000** | 72 |
| c2 | 0.9306 | 58 | **0.9676** | 64 |
| c3 | 0.8333 | 48 | **0.9375** | 60 |
| c4 | 0.8912 | 51 | **0.9468** | 68 |
| c5 | 0.6898 | 57 | **1.0000** | 72 |
| **mean** | **0.8449 ± 0.0885** | **53.2** | ⭐⭐ **0.9734 ± 0.0247** | ⭐⭐ **68.0** |

> ⭐⭐⭐ **Δ = +0.1285, paired 6/6, and the pairwise model's *variance across corners collapses*
> (sd 0.0885 → 0.0247).** ⛔ 11.3 MB — 24× `knn_proto`'s ledger — so it is not an arm. But it settles the
> scientific question: **the ceiling arm is not near a ceiling. It is additive, and this corner-generalisation
> task is a second-order problem.** Two of the six corners are solved *exactly* (72/72 distinct tuples, 1.0000).

⇒ **The gap is entirely mine: the model exists, and my 24× compression of it into the store's basis is what
loses the +0.13.** ⇒ ⭐ the lever is **basis span and interaction rank per byte**, and that is what DEV-20
(a smooth request basis: 58 additive columns → **23**, 1322 pair columns → **201**) and DEV-21 are for.

**DEV-20 running**, and it changes the arithmetic materially. `head = (n_add·K·M + n_pair·r + r·K·M)·4`:

| | one-hot design | ⭐ smooth design |
|---|---|---|
| additive columns | 58 | **23** |
| pair columns | 1,322 | **201** |
| best byte-legal point | span 832, rank 16 | ⭐ **span 1,280–2,048, rank 24–46** |

The smooth design encodes the factors as what they physically are — the three hues on a **circle** (2 Fourier
harmonics), `scale` and `orientation` as **ordered** (degree-3/4 polynomials), and ⛔ `shape` left one-hot
because it is genuinely categorical. ⭐ It should also *extrapolate* into an unseen cell rather than
interpolating a table of cells, which is what a compositional hold-out asks for.

## 2026-09-16T22:50Z — ⛔ DEV-20 (fully smooth basis) REJECTED, and ⭐ a §C.2 finding that matters more

**DEV-20, six configurations × six corners, held-out corners c3–c5** (`knn_proto` there = 0.8025):

| config (smooth additive + smooth pairs) | HELD-OUT modes | tuples | atoms | pca | random | **mean (§C.2)** | Δknn |
|---|---|---|---|---|---|---|---|
| K64 M20 r32 span1280 | 0.7608 ± 0.050 | 52.0 | 0.7639 | 0.7307 | 0.7670 | 0.2994 | −0.042 |
| K128 M12 r24 span1536 | 0.7207 ± 0.040 | 47.0 | 0.7546 | 0.7639 | **0.8017** | **0.4444** | −0.082 |
| K32 M26 r32 span832 | 0.7431 ± 0.051 | 45.3 | 0.7361 | 0.7531 | 0.6674 | 0.2114 | −0.059 |
| K128 M10 r32 span1280 | 0.6898 ± 0.045 | 51.7 | 0.7176 | 0.7569 | 0.7269 | **0.4961** | −0.113 |
| K64 M20 r16 span1280 | 0.7176 ± 0.020 | 52.0 | 0.7593 | 0.6613 | 0.7269 | 0.2870 | −0.085 |

⛔ **Rejected.** Every configuration is below DEV-16's one-hot design (held-out **0.7971**), and at several
of them the §C.1 atom floor now **wins** (Δatoms negative). ⇒ replacing the ADDITIVE one-hot block with a
smooth basis loses real resolution. The additive block is `knn_proto`'s own class and it needs all 58 columns.

### ⭐⭐ But the `mean` column is the important finding, and it is a METHOD result
The §C.2 laundering control (every mode replaced by its well mean) **stops collapsing as K grows**:

| K | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| §C.2 `mean` control | 0.20 | **0.21 – 0.25** | 0.29 – 0.30 | ⛔ **0.44 – 0.50** |

⛔ At `K = 128` the store's CONTENT is gone and the arm still scores **0.50** — because with a linear read,
`K` wells give `K` free scalars, so the **PARTITION alone** supplies 128 degrees of freedom.
⭐ **That is a genuine trap for any linear/soft read over a fine partition, and the shipped §C.2 control
catches it exactly.** ⇒ **K ≳ 128 is disqualified for this read**, not because it scores badly but because its
control does not collapse. I am treating that as a hard filter from here: ⛔ **no configuration whose §C.2
control clears the constant floor may be reported as an arm.** (At K ≤ 64 the control sits at 0.21–0.30, far
below the 0.5278 constant floor. ✅)

⇒ DEV-21 (K = 256/512, full-span) is **cancelled before running**: by this filter its control cannot collapse
— `K` free scalars is already more freedom than the task needs. ⛔ Not run, and recorded as not run.

### ⭐ DEV-23, launched: the HYBRID design
Keep the additive block **one-hot at full resolution (58 columns)** — DEV-20 says it needs to be — and make
only the **pair** block smooth (**201 columns instead of 1322**), since the pair block is the one that is
factorised and therefore the one whose width the budget actually pays for. That buys, all under 475,136 B:

| K | M | r | span | head B | store B | total B |
|---|---|---|---|---|---|---|
| 64 | 16 | **21** | 1024 (50 %) | 340,468 | 131,072 | **471,540** |
| 32 | 20 | **48** | 640 (31 %) | 309,952 | 163,840 | **473,792** |
| 64 | 14 | **34** | 896 (44 %) | 357,064 | 114,688 | **471,752** |
| 32 | 26 | 16 | 832 (41 %) | 259,136 | 212,992 | 472,128 |
| 16 | 32 | **33** | 512 (25 %) | 212,900 | 262,144 | 475,044 |

⭐ Against DEV-16's best (rank 16 over **2** pairs = 160 one-hot columns), this is rank **21–48** over **all
15** pairs. ⛔ And it is the last capacity idea I have; if it does not clear `knn_proto` on the held-out
corners, the honest answer for this hub is a measured tie-or-loss against the ceiling.

## 2026-09-16T23:10Z — ⛔ DEV-23 (hybrid design) also rejected. **The search is over; DEV-16's configuration stands.**

Hybrid = one-hot additive (58 columns) + **smooth** pair block (201 columns), rank 21–48, six corners,
held-out c3–c5 (`knn_proto` = 0.8025):

| config | HELD-OUT modes | tuples | atoms | pca | random | mean (§C.2) | Δknn | Δatoms |
|---|---|---|---|---|---|---|---|---|
| K64 M16 r21 ridge10 | 0.7677 ± 0.027 | 46.7 | 0.7809 | 0.7238 | 0.7423 | 0.2585 | −0.035 | −0.013 |
| K64 M14 r34 | 0.7747 ± 0.027 | 40.3 | 0.8094 | 0.7353 | 0.6350 | 0.3164 | −0.028 | −0.035 |
| K32 M26 r16 | 0.7770 ± 0.049 | 37.3 | 0.7099 | 0.7076 | 0.6798 | 0.2454 | −0.026 | **+0.067** |
| K64 M16 r21 ridge100 | 0.7623 ± 0.038 | 46.3 | 0.7253 | 0.7608 | 0.7083 | 0.2623 | −0.040 | +0.037 |
| K32 M20 r48 | 0.7245 ± 0.015 | 36.3 | **0.8310** | 0.7384 | 0.5432 | 0.2469 | −0.078 | −0.107 |
| K16 M32 r33 | 0.7338 ± 0.017 | 32.0 | 0.7276 | 0.6620 | 0.5687 | 0.1937 | −0.069 | +0.006 |

⛔ Rejected: every row is below DEV-16's held-out **0.7971**, `Δknn` is negative everywhere and **1/3** paired
wins everywhere. ⚠ And the smooth pair basis costs distinct tuples badly (32–47 vs DEV-16's 55.3).

### ⛔ Everything I tried, and what happened — the complete list
| parameterisation | held-out result | verdict |
|---|---|---|
| one-hot additive + **one-hot top-2 pairs**, rank 16 | **0.7971 (55.3 tuples)** | ⭐ **best; Δknn −0.0054** |
| one-hot additive + all 15 one-hot pairs, rank 4–12 | 0.713–0.783 | ⛔ rank truncation over 1322 columns |
| smooth additive + smooth pairs | 0.690–0.761 | ⛔ additive block needs full resolution |
| one-hot additive + smooth pairs (hybrid) | 0.724–0.777 | ⛔ ditto, and it costs distinct tuples |
| full-span (K=256/512) | ⛔ **not run** | disqualified: §C.2 control cannot collapse |
| low-rank of the WHOLE head | 0.20–0.67 | ⛔ dead (DEV-15) |
| decoder-diagonal-weighted codebook | null | ⛔ dead (DEV-04) |
| vertex (argmax) read, K-sweep | ≤0.6921 | ⛔ far below the linear read |
| launch-then-snap read | ≤ head_alone always | ⛔ the store can only lose |

⇒ **I am stopping the search.** The configuration is
**`K=32 · M=26 · rank=16 · pairs=(floor_hue×scale, object_hue×scale) · ridge=10`**,
store **212,992 B** + head **259,136 B** = **472,128 B** (0.994× `knn_proto`'s 475,136 B).

**DEV-24 launched:** that ONE configuration, through the **production objects**
(`GenStore` · `gen_factored_read.fit_factored_head` · the shipped `clu_wide_vector` ·
`assert_request_only_factored` · `assert_no_bypass`), on **twelve** corners, with all five bases, plus the
shipped **argmax** read on the *same* store for the like-for-like column. ⛔ Nothing is selected in it.

- ⚠ **Correction (new bullet):** the arm's head is **256,512 B**, not the 259,136 B I wrote at 23:10 — I
  mis-added `58·K·M + n_pair·r + r·K·M = 48,256 + 2,560 + 13,312 = 64,128` floats. **Total = 212,992 + 256,512
  = 469,504 B = 0.988 × `knn_proto`'s 475,136 B.** ⭐ The figure is now taken from
  `FactoredRequestHead.byte_ledger()` on the LIVE object, ⛔ never from the formula. PREREG updated.

## 2026-09-16T23:41Z — ⭐⭐ DEV-24: the arm, through the PRODUCTION objects, on TEN corners

`K=32 · M=26 · rank=16 · pairs=(floor_hue×scale, object_hue×scale) · ridge=10` ·
store **212,992 B** + head **256,512 B** = ⭐ **469,504 B = 0.988 × `knn_proto`'s 475,136 B** ·
channel **832 reals** (`knn_proto`'s is 2048).
⭐ Built from `GenStore` + `gen_factored_read.fit_factored_head` + the **shipped**
`gen_compose.clu_wide_vector`. **Assertions PASS:** `assert_request_only_factored`
(independent re-implementation of the read agrees to `1.67e-06`, float32 rounding; provenance re-derived;
bit-identical under feature replacement) and the shipped `assert_no_bypass` on the same store object.

⭐ **The eval_a guard fired and did its job:** 2 of the 12 corners were **REFUSED** because they intersected
`eval_a` by one row each. ⛔ They were **not trimmed** — a trimmed corner is not a Montero-shaped cell — and
are recorded as skipped. **n = 10 corners.**

| row (10 corners, seed 0) | aggregate | **distinct tuples / 72** | deployed bytes |
|---|---|---|---|
| §C.2 store-removed (modes → well mean, same coefficients) | **0.2843** | — | 469,504 |
| best constant output (structural **and** realised) | 0.5278 | **1** | 0 |
| random-basis substitute (D-matched null) | 0.6488 | — | 469,504 |
| ⭐ **the shipped ARGMAX read on the SAME store** | **0.6206** | **20.7** | 469,504 |
| PCA-basis substitute | 0.7042 | — | 469,504 |
| ⭐ **atom-basis §C.1 floor** (M real train rows/well) | **0.7921** | — | 469,504 |
| ⭐⭐ **`knn_proto`** | **0.8396 ± 0.0619** | **49.1** | **475,136** |
| ⭐⭐ **THE ARM — factored dictionary read** | ⭐⭐ **0.8468 ± 0.0633** | ⭐ **52.0** | ⭐ **469,504** |

**Paired per-corner differences, with proper intervals:**

| comparison | mean Δ | sd | SE | 95 % CI | paired wins |
|---|---|---|---|---|---|
| arm − **`knn_proto`** | **+0.0072** | 0.0488 | 0.0154 | **[−0.023, +0.037]** | **6/10** |
| arm − **atom basis (§C.1)** | ⭐ **+0.0546** | 0.0623 | 0.0197 | ⭐ **[+0.016, +0.093]** | **8/10** |
| arm − PCA basis | ⭐ **+0.1426** | 0.1313 | 0.0415 | **[+0.061, +0.224]** | 8/10 |
| arm − **argmax read on the same store** | ⭐⭐ **+0.2262** | 0.0715 | 0.0226 | ⭐⭐ **[+0.182, +0.271]** | ⭐⭐ **10/10** |

### ⭐ What this is, stated exactly
1. ⭐⭐ **Against `knn_proto` it is a TIE, not a win.** `Δ = +0.0072 ± 0.0154`; the 95 % CI straddles zero and
   6/10 corners go the right way. ⛔ **I will not call +0.0072 a win.** It is inside the task's `m = 0.020`
   band, so *if* it reproduced with 3/3 same sign on B it would be a **TIE (favourable)** by the ceiling
   rule — ⚠ but with a per-corner sd of 0.049 that sign consistency is close to a coin flip, and I say so.
2. ⭐⭐ **§C.1 PASSES with a CI clear of zero: +0.0546 [+0.016, +0.093], 8/10.** At identical bytes and
   identical configuration, a dictionary of **learned modes** beats a dictionary of **real stored rows**.
   ⇒ **the store is not a lookup table, and at these bytes it is no longer decorative.**
3. ⭐ **§C.2 PASSES hard: 0.2843**, far below the 0.5278 constant floor.
4. ⭐ The PCA and random substitutes lose by +0.143 and +0.198 ⇒ ⛔ the store is not "any 832-dimensional
   subspace", and the partition is not free.
5. ⭐⭐ **On the SAME store, the shipped ARGMAX read scores 0.6206 with 20.7 distinct tuples; the factored
   read scores 0.8468 with 52.0 — +0.2262 [+0.182, +0.271], 10/10 corners.**
   ⇒ **the read verb was worth more than everything else in this run put together.**

⭐ The pre-registered prediction (written at 23:10, before this) was `Δ = −0.03`, 90 % interval
`[−0.10, +0.02]`. The realised DEV estimate `+0.0072` sits **inside** that interval. ⛔ The PREREG is not
edited; the outcome is appended to it, dated.

# ⛔⛔⛔ 2026-09-16T23:44Z — **BLOCKER: THE SCORED CELL CANNOT RUN ON THIS MACHINE.** Measured, not inferred.

While extending the DEV validation to seeds 1 and 2 I hit the pinned harness's own re-encode gate. I then
measured it deliberately on **all three seeds** (`$CHLU_OUT/dev/exp26_reencode.py`, full strength):

| seed | cache scope | cache rows | encoder hash matches the cache manifest | **`_assert_cache_reencodes`** | max \|Δ\| | ULP of the cache dtype (bound **1**) | elements differing (bound **1 %**) |
|---|---|---|---|---|---|---|---|
| 0 | `full` | 480,000 | ✅ True | ⛔ **FAIL** | 0.001953125 | **51.0** | 1280 / 131072 = **0.977 %** |
| 1 | `used` | 9,288 | ✅ True | ⛔ **FAIL** | 0.001953125 | **81.0** | 640 / 131072 = **0.488 %** |
| 2 | `used` | 9,288 | ✅ True | ⛔ **FAIL** | 0.001953125 | **59.0** | 1231 / 131072 = **0.939 %** |

⭐ `encode precision: requested=highest resolved=HIGHEST backend=cpu` — the pin's precision fix is active.
⭐ The **encoder parameter hash matches the cache manifest on every seed**, so this is not a wrong checkpoint.
⭐ `max |Δ| = 2⁻⁹ = 0.001953125` on all three seeds; the differing elements sit around 0.03–0.12, so they move
by **2–6 % relatively**. ⛔ That is a different computation, not a rounding step.
⚠ The *fraction* differing is inside the 1 % bound on all three seeds. It is the **magnitude** (51–81 ULP
against a bound of 1) that fails. The bundle was `built_on_host: batMac.local`; the caches are Mac-written.

### Why this blocks the scored run — three facts, each checked
1. ⛔ **`chlu/experiments/exp_gen_cell.py:1780`** (`run_seed`) calls
   `_assert_cache_reencodes(dataset, config, encoder, crows, feats_cache, cman, log=tlog)` — ⭐ **with no
   `will_extend`**, and the signature's default is `will_extend: bool = True` ⇒ **ENFORCED** ⇒ the call
   **raises** on this box. It sits **before** `gather`, before any arm and **before the decoder ceiling**, so
   the cell aborts having computed **no held-out number at all**.
2. ⛔ Even if that preflight were scoped, **seeds 1 and 2 would still fail at the point of extension.**
   Measured coverage of what the cell actually needs (`pool ∪ eval_b ∪ eval_a` = 33,864 rows):
   | seed | inside the cache | rows this box would have to ENCODE |
   |---|---|---|
   | 0 | **33,864 / 33,864 = 100.0 %** | ⭐ **0** |
   | 1 | 1,665 = 4.9 % | ⛔ **32,199** |
   | 2 | 1,623 = 4.8 % | ⛔ **32,241** |
   `gather_features` re-runs the gate at **full strength at the point of extension** (`exp_gen_store.py:284`),
   and it would fail. ⇒ ⛔ **seeds 1 and 2 are impossible here, full stop.**
3. ⛔ Seed 0 is the only seed that *could* run — it needs **zero** new rows — but only if `run_seed`'s
   preflight were called as `will_extend=False` with a reason, the sanctioned pattern the gate's own docstring
   describes and `exp_gen_wells_route_b` already uses. ⭐ **That is a one-line change to a pinned harness file,
   and charter §A.4 says a hub that edits the harness voids its numbers. ⛔ So I did not make it.**

### ⇒ The verdict, and what I did about it
> ⛔ **The 3-seed scored cell is not executable on FORGIS_14 as pinned.** Not by me, and not by
> `hub/refusal` or `hub/composition` either — this is a property of the bundle and the box, not of my arm.

⛔ COMMON §0 is explicit that its override covers **exactly one named test** and *"is not a licence to wave
through any other red check"*. This is another red check. ⛔ **I have not disabled it, not widened it, not
passed `will_extend=False` from my own code to sneak past it, and not edited any pinned file.** It is
reported as the blocker it is.
⭐ **And the gate is CORRECT to fire.** COMMON §4's hard constraint is *"never mix Mac-produced and
Windows-produced artefacts inside one comparison"*. Seeds 1 and 2 would need **32,000 Windows-encoded rows**
fed into a table whose rival rows are Mac-produced. ⭐ **The pin's own gate is enforcing COMMON §4, in code.**

### ⭐ The one piece of good news, and it is load-bearing for everything above
**Seed 0's cache covers 100 % of what the cell needs — and 100 % of every DEV corner I used (3,456 distinct
rows across all 48 candidate corners).** ⇒ ⭐⭐ **Every number in this log was computed on the MAC-WRITTEN
cached features. This hub re-encoded nothing, ever.** The DEV work is not contaminated by the discrepancy it
just found; it is the reason I can trust it.

### What the harness's owner needs (⛔ not mine to apply)
1. `exp_gen_cell.run_seed:1780` → scope the preflight the way the pin already supports:
   `_assert_cache_reencodes(..., will_extend=(n_shortfall > 0), reason="coverage gate: the cell's row set is
   fully inside the cache")`, with `n_shortfall` computed from `crows` two lines above — the
   `exp_gen_wells_route_b` pattern, so the two can never disagree. ⭐ That unblocks **seed 0** on any machine.
2. Seeds 1 and 2 need their bridge caches re-issued at `cache_scope="full"` **from the Mac**, or the whole
   3-seed cell has to run on the Mac. ⛔ There is no correct way to produce them here.
3. ⚠ Worth knowing regardless: the encoder does not reproduce its own cache **on seed 0's `full` cache either**
   (51 ULP). So the gate would fail on this box for *any* extending run, on *any* seed.

## 2026-09-16T23:45Z — code committed to the branch, additive only

`cb6d4d4  [hub-generation-table] the factored dictionary read, additive only` on `hub/generation-table`,
off `8e36bc9`. **5 new files, 940 insertions, 0 deletions, 0 modified files.**
Verified from the MAIN repo (`git -C $CHLU_ROOT log --oneline main..hub/generation-table`) so the shared ref
really did advance — the w4 lesson in `AGENT_PROTOCOL` §3.2.

⭐ **Every pinned harness file is byte-identical to the pin**, checked one by one:
`exp_gen_cell.py` · `exp_gen_store.py` · `gen_compose.py` · `gen_store.py` · `gen_dream.py` ·
`gen_request.py` · `gen_probe.py` · `gen_rowsplit.py` · `gen_landscape.py` — ✅ all unchanged.

| new file | what it is |
|---|---|
| `chlu/core/gen_factored_read.py` | the read under test + `assert_request_only_factored` + the byte and channel ledgers |
| `chlu/core/gen_partition.py` | the equal-well correlation partition (same `\|corr\|`, same ward linkage, cut by leaf order) |
| `chlu/core/gen_address.py` | the shipped naive-Bayes head re-estimated discriminatively **at identical bytes** |
| `chlu/core/gen_sensitivity.py` | `diag(JᵀJ)` of the frozen decoder — ⛔ a **recorded dead end**, kept because it is a measurement |
| `tests/test_gen_factored_read.py` | **8 tests, all passing** — the read is a pure weighted mode sum; it takes no features; §C.2 moves it; the byte ledger equals its closed form; `steps=0` is bit-identical to the shipped estimator; the partition is a partition |

# ⛔⛔ 2026-09-17T00:13Z — **THE 49-CORNER NUMBER, AND IT OVERTURNS MY 23:41 ENTRY**

I widened DEV-24 from 10 corners to **every Montero-shaped corner the grid admits**. 64 candidates,
**15 REFUSED** by the guard (they intersect `eval_a`/`eval_b` or do not realise 72 rows — ⛔ refused, never
trimmed), **n = 49 scored**. Same fixed configuration, same production objects, nothing re-selected.

| row (49 corners, seed 0) | aggregate | sd | **distinct tuples / 72** | deployed bytes |
|---|---|---|---|---|
| §C.2 store-removed (modes → well mean) | **0.2860** | 0.0981 | **8.7** | 469,504 |
| best constant output | 0.5278 | — | **1** | 0 |
| ⛔ **the shipped ARGMAX read, same store** | **0.6314** | 0.0719 | **18.9** | 469,504 |
| random-basis substitute | 0.6623 | 0.0792 | 33.4 | 469,504 |
| PCA-basis substitute | 0.7055 | 0.1149 | 48.7 | 469,504 |
| ⭐ **atom-basis §C.1 floor** | **0.7954** | 0.0806 | 51.9 | 469,504 |
| ⭐⭐ **THE ARM** | **0.8135** | 0.0647 | **49.9** | **469,504** |
| ⭐⭐ **`knn_proto`** | **0.8307** | 0.0621 | 51.8 | **475,136** |

**Paired per-corner differences (n = 49):**

| comparison | mean Δ | SE | 95 % CI | paired wins | t |
|---|---|---|---|---|---|
| ⛔ arm − **`knn_proto`** | **−0.0172** | 0.0089 | **[−0.0347, +0.0003]** | **16/49** | −1.93 |
| ⚠ arm − **atom basis (§C.1)** | **+0.0180** | 0.0089 | **[+0.0005, +0.0356]** | 28/49 | +2.02 |
| ⭐ arm − PCA basis | +0.1079 | 0.0178 | [+0.073, +0.143] | 39/49 | +6.05 |
| ⭐ arm − random basis | +0.1512 | 0.0146 | [+0.123, +0.180] | 44/49 | +10.4 |
| ⭐⭐ arm − **§C.2 store-removed** | **+0.5275** | 0.0163 | [+0.496, +0.560] | **49/49** | +32.3 |
| ⭐⭐ arm − **the shipped ARGMAX read on the SAME store** | **+0.1821** | 0.0107 | **[+0.161, +0.203]** | **49/49** | **+17.0** |
| arm − best constant | +0.2857 | | | | |

## ⛔⛔ THE CORRECTION, stated plainly
> **My 23:41 entry called this a TIE on the strength of 10 corners (Δ = +0.0072, 6/10). On 49 corners it is
> `Δ = −0.0172`, 16/49, t = −1.93. ⛔ THAT IS A LOSS, NOT A TIE.**
> The ceiling rule requires `0 < Δ ≤ m` for a favourable tie; ⛔ **the sign is negative, so this is a LOSS
> against `knn_proto` however small the magnitude is.**

⭐ The first ten corners were a favourable draw, and one of them (`c0`) was the corner the pair set had been
selected on. **This is the maximum-over-a-search bias, one level up: I had already corrected for it once at
21:49 and it was still there at n = 10.** ⛔ Recorded as the second such correction in this run, not buried.

⚠ **And §C.1 went from comfortable to marginal:** `+0.0546 [+0.016, +0.093]` at n = 10 became
`+0.0180 [+0.0005, +0.0356]` at n = 49. The CI still excludes zero, t = 2.02, **but only just, and only
28/49 corners go the right way.** ⇒ ⭐ **I will report §C.1 as PASSED-BUT-MARGINAL and ⛔ not as a clean pass.**

## ⭐⭐ What is left standing, and it is large and not marginal at all
| claim | Δ | 95 % CI | paired |
|---|---|---|---|
| ⭐⭐ **the factored read beats the shipped ARGMAX read on the SAME STORE, SAME BYTES** | **+0.1821** | [+0.161, +0.203] | **49/49** |
| ⭐⭐ **the store is load-bearing (§C.2)** | **+0.5275** | [+0.496, +0.560] | **49/49** |
| ⭐ the store is not "any subspace" (PCA) | +0.1079 | [+0.073, +0.143] | 39/49 |
| ⭐ the partition is not free (random basis) | +0.1512 | [+0.123, +0.180] | 44/49 |
| ⭐ it clears the constant-output floor | +0.2857 | — | — |
| ⭐ distinct tuples | **49.9 / 72** against the argmax read's **18.9** and the constant's **1** | | |

⭐ **The read verb is worth +0.18 with 49/49 paired wins and t = 17.** That is the result of this hub, and it
is a bigger, better-supported effect than anything the ceiling comparison was ever going to give.

## 2026-09-17T00:31Z — ⭐ DEV-27: the §C.1 floor, made as hard as I can make it

DEV-24's atom control picks `M` train rows **uniformly at random** — the easy version of "store real rows".
With the §C.1 margin only `+0.0180` at n = 49, the choice of atom selector could decide the verdict, so I
measured three selectors at identical bytes and identical head configuration, n = **24** corners:

| dictionary (identical bytes, identical head) | aggregate | tuples | **Δ (arm − this)** | SE | 95 % CI | wins | t |
|---|---|---|---|---|---|---|---|
| ⭐ **the arm — learned k-means centroids** | **0.8146** | 48.4 | — | | | | |
| `atoms_kpp` — M maximally-spread real rows (k-means++ on the rows) | 0.7924 | 49.3 | ⚠ **+0.0222** | 0.0134 | ⚠ **[−0.0041, +0.0485]** | 17/24 | +1.65 |
| `atoms_random` — M uniformly random real rows (DEV-24's control) | 0.7847 | 48.6 | **+0.0299** | 0.0101 | **[+0.0100, +0.0498]** | 16/24 | +2.95 |
| ⭐⭐ `atoms_medoid` — for each learned cluster, the real row NEAREST its centre | 0.7296 | 45.1 | ⭐⭐ **+0.0850** | 0.0148 | ⭐⭐ **[+0.0560, +0.1140]** | **23/24** | **+5.75** |

**Reading it honestly, all three ways:**
- ⭐⭐ **`atoms_medoid` is the scientifically sharp test** — it differs from the arm in **exactly one thing**:
  whether a dictionary entry is the **fitted centre** of a cluster or the **real stored row nearest to it**.
  Same partition, same clustering, same count, same bytes. **+0.0850 [+0.056, +0.114], 23/24, t = 5.75.**
  ⇒ ⭐ **Fitting the dictionary buys a real and large amount over storing rows. The store is not a lookup table.**
- ✅ Against uniformly random atoms: **+0.0299**, CI excludes zero.
- ⚠ **Against the most adversarially-chosen atoms (`atoms_kpp`), the margin is +0.0222 and the 95 % CI
  INCLUDES ZERO.** ⛔ **So I will not claim a clean §C.1 pass against every possible atom selector.** I claim
  it against the random control and against the medoid control, and I report the `kpp` result as it is.

# ⛔⛔ 2026-09-17T00:47Z — DEV-28: **MY PCA CONTROL WAS UNFAIR, AND FIXING IT RETRACTS A CLAIM**

The `pca` substitute basis I used in DEV-24/27 was `Vt[:M]` — the top `M` **centred** principal directions of
each well. ⛔ **That basis does not contain the well's mean**, so a read over it cannot represent the mean at
all, while the arm's k-means centroids are uncentred and carry the mean in every one of them. ⇒ the
`+0.1079` I reported over "PCA" was **partly an artefact of my own control**.

The fair control at identical bytes is the optimal `M`-dimensional **affine** subspace:
`pca_affine = [ well mean ] + top (M−1) centred principal directions`. Measured on 24 corners:

| dictionary (identical bytes, identical head) | aggregate | tuples | Δ (arm − this) | SE | 95 % CI | wins | t |
|---|---|---|---|---|---|---|---|
| ⭐⭐ **`pca_affine` — mean + M−1 PCA directions** | ⭐⭐ **0.8365** | **56.5** | ⛔ **−0.0219** | 0.0196 | [−0.0603, +0.0165] | **6/24** | −1.12 |
| the arm — learned k-means centroids | 0.8146 | 48.4 | — | | | | |
| `pca_centred` — ⛔ my old, unfair control | 0.7832 | 55.8 | +0.0314 | 0.0213 | [−0.0104, +0.0733] | 16/24 | +1.47 |
| `atoms_medoid` — §C.1, the sharp form | 0.7296 | 45.1 | ⭐ **+0.0850** | 0.0148 | **[+0.056, +0.114]** | **23/24** | +5.75 |
| `mean` — §C.2 | 0.2743 | 9.5 | ⭐⭐ **+0.5403** | 0.0215 | [+0.498, +0.582] | **24/24** | +25.2 |

## ⛔ RETRACTION
> **"The arm beats a PCA basis by +0.1079 [+0.073, +0.143], 39/49 ⇒ the store is not just any
> 832-dimensional subspace" (LOG 00:13Z) is WITHDRAWN.** With the mean direction restored, the PCA basis
> scores **0.8365** and the arm scores **0.8146**: ⛔ **`Δ = −0.0219`, the arm loses on 18 of 24 corners.**
> ⭐ **A per-well AFFINE PCA dictionary is as good as, or better than, the learned k-means dictionary at
> identical bytes** — and it renders **more** distinct tuples (56.5 vs 48.4).

⭐ **What survives, and what it now means:**
- ✅ **§C.1 as the charter actually defines it** — *"the MATCHED-BYTES LOOKUP (atom-mean / kNN on the same
  atoms)"*, i.e. a dictionary of **stored rows** — still passes: **+0.0850 [+0.056, +0.114], 23/24** against
  medoids, **+0.0299 [+0.010, +0.050]** against random atoms. ⚠ **+0.0222, CI [−0.004, +0.049]** against
  maximally-spread atoms.
- ✅ **§C.2** passes overwhelmingly: **+0.5403, 24/24, t = 25.2.**
- ⛔ **But the honest generalisation is narrower than I wrote at 00:13Z:** *learning the dictionary beats
  storing rows; it does **not** beat taking the principal subspace.* ⭐ And a per-well affine PCA basis is
  **not charter-legal as a store** (§B.1 wants atoms that shape an energy, and basins), so it is not an arm —
  ⛔ but it is the right thing for a reviewer to ask about, and the answer is now in the log rather than
  absent from it.
- ⭐⭐ **It is also a design signal for the programme:** at 469,504 B the best byte-matched dictionary found in
  this run is `pca_affine` at **0.8365 with 56.5/72 tuples** — ⭐ *above* `knn_proto`'s 0.8307/51.8 on the
  49-corner measure, though on a different corner subset and ⛔ not a like-for-like claim.

⛔ **This is the third correction I have had to make to my own claims in this run** (19:53Z `quant_ceiling`;
00:13Z the 10-corner "tie"; now the PCA control). Each was found by building the harder control rather than
the easier one. ⭐ I would rather publish three retractions than one unearned number.

---

# ⭐⭐ FINAL REPORT — `hub/generation-table` (charter §E.3: the last section of this same log)

**START_UTC `2026-09-16T19:10:50Z` · this section written `2026-09-17T00:50Z` · elapsed 5 h 39 m of the 12 h
window · deadline `2026-09-17T07:10:50Z`.**

## VERDICT
> ## ⛔ **NOT SOLVED — truthfully, and with the reason measured rather than guessed.**
> **Scored runs spent: 0 of 3.** ⛔ Blocked by **two independent hard constraints**, neither worked around:
> COMMON §1c (the scored-run budget was never given) and the pin's own **re-encode gate, which FAILS on this
> machine on all three seeds**, aborting `exp_gen_cell` before it computes any held-out number.
> On 49 held-out DEV corners the arm I built **loses to `knn_proto` by −0.0172 (16/49, t = −1.93)**, so
> ⛔ **falsifier 1 fails at the ceiling clause** — and I predicted that in writing, before measuring it.

## THE THREE FALSIFIERS
| # | falsifier | verdict |
|---|---|---|
| 1 | beats the five rivals **and** WIN-or-TIE vs `knn_proto` | ⛔ **NOT ANSWERED on `eval_b`** (0 scored runs). On DEV the arm beats the shipped argmax read **+0.182 (49/49)**, the §C.2 control **+0.528**, the random basis **+0.151** and the random-atom §C.1 floor **+0.018** — and **LOSES to `knn_proto` −0.0172 (16/49)**. Negative sign ⇒ **LOSS**, not a tie. |
| 2 | clears the constant-output floor, tuples ≫ 2.67/72 | ⭐ **PASSED on DEV, and not narrowly: 0.8135 with 49.9 distinct tuples of 72**, against the floor's **0.5278 with 1** and the shipped argmax read's **18.9**. (⭐ I had to *raise* that floor first: it is **0.5278**, not 0.3272.) |
| 3 | the store is load-bearing; `store_removed` collapses it | ⭐ **PASSED, hard: 0.8135 → 0.2860 (49.9 → 8.7 tuples), Δ +0.5275, 49/49, t = 32.3**, collapsing far below the constant floor. ⭐ And not load-bearing *only* by construction: the **learned** dictionary beats a **stored-row** dictionary of identical size by **+0.085 (23/24)** against medoids. ⚠ It does **not** beat an affine-PCA dictionary (−0.022) — retracted at 00:47Z. |

## WHAT THE PROGRAMME GETS OUT OF THIS RUN, WHATEVER HAPPENS TO MY ARM
1. ⭐⭐ **The free floor on composition-B is 0.5278, not 0.3272** — structural *and* realised. **Six of the ten
   published rows lose to emitting one picture 72 times.**
2. ⭐⭐⭐ **`knn_proto` is not a ceiling; it is the best ADDITIVE request model.** Adding pairwise factor terms
   reaches **0.9734 ± 0.0247 (68.0/72 tuples)** against its **0.8449 ± 0.0885 (53.2)** — **+0.1285, 6/6
   corners**, two corners solved exactly. ⛔ At 11.3 MB it is a bound, not an arm. **The table has been
   measuring an additive ceiling against arms asked to be compositional.**
3. ⭐⭐⭐ **The shipped 30-bit read is one VERTEX of a subspace its own modes already span.** On the *shipped*
   store: vertex **0.4884**, convex hull **0.5509**, **linear span 0.8264**. The read verb, not the codebook
   and not the addresser, is the largest identified loss — and changing it is worth **+0.182 with 49/49
   paired wins (t = 17)** at identical bytes.
4. ⭐⭐ **The quantisation loss is structure, not magnitude:** isotropic noise at the same MSE costs ~0.03
   where the codebook's own residual costs up to **0.43**. ⇒ ⛔ the codebook is not at an information limit,
   and a **diagonal** decoder metric cannot fix it (measured null).
5. ⛔⛔ **The scored cell is unrunnable on FORGIS_14 for every hub**, with the numbers and the one-line remedy
   in the 23:44Z entry.
6. ⭐ **Two methodological controls the programme should adopt:** a PCA substitute basis must be **AFFINE**
   (a centred one is unfairly weak — it cost me a claim); and a linear/soft read over `K` wells gets `K` free
   scalars from the **partition** alone, so **§C.2 must be reported with its `K`** (the control stops
   collapsing at `K ≳ 128`).

## DELIVERABLES
| | |
|---|---|
| running log | **this file**, appended throughout, ⛔ never written at the end |
| pre-registration | `PREREG-hub-generation-table-1.md` — bar and prediction committed **before** any held-out number; **not executed**; outcome and blocker appended as dated appendices, body untouched |
| results ledger | `RESULTS-LEDGER.md` — **0 scored runs**, with both reasons |
| one-page report | `.claude/outputs/hub-generation-table.md` |
| code | branch `hub/generation-table` @ `cb6d4d4` — **5 new files, 940 insertions, 0 deletions, 0 modified**; 8 tests passing; every pinned harness file verified byte-identical to the pin |
| raw DEV data | `.claude/outputs/hub-generation-table/dev/*.json` (28 experiments) + their logs |

## COST
**5 h 39 m wall clock.** Local CPU only, **5 threads**, per COMMON §3 / AMENDMENT 1.
⛔ **No GPU. No GCloud. Nothing spent. CSF3 NOT RUN (2FA — unreachable by any hub).**
Sanctioned additive environment work: `h5py 3.16.0` (a declared `gen` extra; numpy stayed 2.4.1) and
`3dshapes.h5` downloaded to the declared data root in **129 s, SHA-256 verified against the shipped constant**.

## ⚠ THE QUESTION STILL OPEN TO THE OPERATOR
> **How many of the 3 scored runs for target `generation-table` remain unspent across all machines?**
> Asked at 19:10Z, the first entry in this log. ⛔ Unanswered. Everything is pre-registered and ready —
> ⚠ **but even with an answer, blocker 2 means the cell cannot run here as pinned.**

## ⭐ WHAT I WOULD DO NEXT, IN ORDER
1. Fix the harness blocker (one line, seed 0) or re-cut seeds 1–2's caches at `full` scope **on the Mac**.
2. ⭐ **Re-score the shipped `clu` arm with the LINEAR read verb.** It is the biggest measured effect in this
   run (+0.182, 49/49) and it needs **no new bytes and no new store** — only `clu_wide_vector` with
   real-valued request-addressed weights instead of `read_from_codes` with an argmax.
3. ⭐ **Re-state the table against 0.5278**, and put the pairwise bound (0.9734) on it as a reference row, so
   the next hub is not sent to beat an additive model believing it is a ceiling.
4. ⛔ **Do not spend another hub on the settling dynamics or on the codebook's capacity.** Both are measured
   here as not the lever.

# ⭐⭐ 2026-09-17T02:30Z — DEV-29: **THE SHIPPED STORE, THE SHIPPED BYTES, ONLY THE READ VERB CHANGED**

The most conservative experiment in the run, and the one my top recommendation rests on. The store is the
**SHIPPED** one — `K = 5` **ward-grouped** wells, `M = 64`, fitted by the **SHIPPED `gen_store.train_store`**
(⛔ not my Lloyd variant) — and the head's additive block alone is `58 · K·M · 4` = `58 · 320 · 4` =
**74,240 B**, ⭐ the shipped addressing head **to the byte**. ⇒ `524,288 + 74,240 = 598,528 B`,
**byte-identical in every column to the shipped `clu` arm.** The ONLY change is
`read_from_codes(store, argmax)` → `clu_wide_vector(store, real-valued request-addressed weights)`.

⚠ This run needed `mode_basis` to accept **ragged** wells — the shipped ward grouping is unequal
(`[303, 305, 573, 501, 366]`). Fixed in `cb86d62`, with two tests (`9f096a3`); a non-uniform **M** is still
refused. ⛔ Without that fix the arm could not touch the shipped store at all, which would have made this
whole comparison impossible.

| row (6 corners, seed 0) | aggregate | sd | **distinct tuples / 72** | deployed bytes |
|---|---|---|---|---|
| §C.2 store-removed (argmax / linear) | 0.0741 / **0.0710** | 0.049 / 0.044 | **1.0 / 1.5** | 598,528 |
| best constant output | 0.5278 | — | **1** | 0 |
| ⛔ **`clu_argmax` — THE SHIPPED ARM** | **0.5490** | 0.1027 | ⛔ **9.5** | **598,528** |
| ⭐⭐ **`linear_additive` — same store, same bytes, linear read** | ⭐⭐ **0.6682** | 0.0587 | ⭐ **35.0** | ⭐ **598,528** |
| §C.1 `atoms_medoid` under the linear read | 0.6755 | 0.0702 | 37.3 | 598,528 |
| ⭐ `linear_pairs16` (+ rank-16 pair terms) | **0.7400** | 0.0737 | **41.3** | ⚠ 629,248 (1.051×) |
| `knn_proto` | 0.7909 | 0.0499 | 53.8 | 475,136 |

| paired difference (n = 6) | mean | SE | 95 % CI | wins | t |
|---|---|---|---|---|---|
| ⭐⭐ **linear − argmax, byte-identical** | **+0.1192** | 0.0450 | **[+0.031, +0.208]** | **6/6** | +2.65 |
| ⭐⭐ **linear+pairs − argmax** (+5 % bytes) | **+0.1910** | 0.0476 | **[+0.098, +0.284]** | **6/6** | +4.01 |
| ⭐⭐ linear − §C.2 store-removed | **+0.5972** | 0.0254 | [+0.547, +0.647] | **6/6** | +23.5 |
| ⚠ **linear − §C.1 medoid atoms** | ⛔ **−0.0073** | 0.0227 | [−0.052, +0.037] | **2/6** | −0.32 |
| ⛔ linear − `knn_proto` | −0.1227 | 0.0224 | [−0.167, −0.079] | 0/6 | −5.49 |
| ⛔ linear+pairs − `knn_proto` | −0.0509 | 0.0370 | [−0.124, +0.022] | 3/6 | −1.38 |

## ⭐⭐ THE ACTIONABLE RESULT
> **On the shipped store, at a byte ledger identical in every column to the shipped `clu` arm, changing the
> read from the 30-bit argmax to a request-addressed LINEAR combination of the same modes is worth
> `+0.1192 [+0.031, +0.208]`, 6/6 corners — and takes the distinct factor tuples from `9.5` to `35.0` of 72.**
> ⭐ With rank-16 pair terms at +5 % bytes: `+0.1910 [+0.098, +0.284]`, 6/6, tuples `41.3`.
> ⛔ **No new store, no new bytes, no new decoder. `gen_compose.clu_wide_vector` already exists and already
> takes no feature argument.**

## ⚠⚠ THE CAVEAT THAT GOES WITH IT, AND IT IS NOT SMALL
⛔ **At `K = 5` the §C.1 medoid floor TIES the linear read (−0.0073, 2/6).** A dictionary of **real stored
rows** does as well as the learned k-means modes on the shipped 5-well store. ⇒ ⭐ **the §C.1 pass I have is
only at `K = 32, M = 26`** (`+0.0850 [+0.056, +0.114]`, 23/24) — ⛔ **it does not hold at the shipped `K = 5`.**
⇒ **the read-verb change is worth taking on its own merits; it does NOT by itself make the shipped
5-well store non-decorative.** Those are two separate claims and I am keeping them separate.

## ⭐ AN EXTERNAL VALIDATION OF THE DEV INSTRUMENT, FOR FREE
`clu_argmax` here reads **0.5490 ± 0.1027 with 9.5 distinct tuples**, against the published B row's
**0.5949 ± 0.0082 with 9.00 ± 0.82 tuples**. ⭐ **The distinct-tuple count — the single most diagnostic
column on this table — reproduces almost exactly**, on a different corner, on a different machine, through my
own wiring. That is the best evidence I have that the DEV instrument is measuring the same object the
published table measures.

## ⚠ HOW THIS RUN ENDED — an environment failure, reported as one
⛔ **DEV-29 died at corner 6 of 12**:
```
allocateMappedMemory failed: The paging file is too small for this operation to complete.
LLVM ERROR: Unable to allocate section memory!     exit=127
```
Host commit-space exhaustion — three hubs sharing the box, and `train_store` re-JITs for every corner because
the ward grouping gives **different well sizes each time**, so XLA compiles a new program per corner (visible
in the corner times: 546 s → 675 → 755 → 939 → 1079 → 1258). ⛔ Not a bug in the science.
⭐ **Six corners were already on disk, because the ledger is written the moment each corner completes.** The
charter's *"a kill costs ONE UNIT, never the run"* is the reason this entry has numbers in it at all.
⛔ **I did not restart it**: the box is resource-constrained, the effect is large and 6/6 consistent, and
spending another hour to move `n` from 6 to 12 would not change the conclusion.

## 2026-09-17T02:31Z — the falsifier-3 statement, tightened after DEV-29
⚠ **Correction to the FINAL REPORT's falsifier-3 row (new bullet; ⛔ the row is not rewritten):** the §C.1
"learning beats storing rows" margin of **+0.085 (23/24)** is measured at **`K = 32, M = 26`**. ⛔ At the
**shipped `K = 5`** the same comparison is **−0.0073, 2/6** — a tie. ⇒ **Falsifier 3 passes on §C.2
everywhere (the control collapses to 0.07–0.29 against a 0.5278 floor), and on §C.1 only at the finer store.**
⛔ I am not claiming the shipped 5-well store is non-decorative; I am claiming the 32-well one is, and that
the read verb is worth +0.12 on the shipped one regardless.

---

# ⭐ 2026-09-17T02:32Z — CLOSING: packaging, verification, and the elapsed-time statement

## Final verification sweep (all ✅)
| check | result |
|---|---|
| any python of mine still running (COMMON §3) | ✅ **none** |
| any chain/waiter script of mine still looping | ✅ **none** |
| worktree clean | ✅ `git status --short` empty |
| branch ref visible **from the MAIN repo**, not just the worktree (AGENT_PROTOCOL §3.2) | ✅ `9f096a3`, `cb86d62`, `cb6d4d4` |
| the ten pinned harness files byte-identical to the pin | ✅ `exp_gen_cell` · `exp_gen_store` · `gen_compose` · `gen_store` · `gen_dream` · `gen_request` · `gen_probe` · `gen_rowsplit` · `gen_landscape` · `gen_bridge` |
| wrote only to `$CHLU_HUB_OUTPUTS/hub-generation-table/**` (+ `hub-generation-table.md`, my declared report path) | ✅ |
| ⛔ nothing deleted (COMMON §5.1) | ✅ — see `RETURN-EXCLUSIONS.json` |

## The git bundle — ⭐ WHOLE BRANCH, ⛔ never a range
`git bundle create … hub/generation-table` → `hub-generation-table.branch.bundle`,
**3,194,185 B**, `sha256 852aef52ea4463747fe2266f502ed9b52e5652a9062c6d3339a00ae4617f6384`.
`git bundle verify` → *"The bundle records a complete history"*, ref
`9f096a3072991d2462c43f1dfd543aa9321eb118 refs/heads/hub/generation-table`.
⭐ Whole-branch as COMMON §6 requires: a thin bundle's prerequisite is this repo's synthetic base commit,
which the Mac has never seen.

## The return pack — and the one judgement call in it, made in the open
`scripts/bundle_hub_workspace.sh --return-pack hub-generation-table` on the real outputs directory
**dry-runs at 187 files / 2,178,765,992 B (2.03 GiB)**. ⛔ That is not a result set; **99.9 % of it is
`dev/cache/**`** — per-corner channel-geometry `.npz`, fitted PQ codebooks, Jacobian probe samples and
gathered pool features, all **regenerable in seconds to a minute** by `dev/fast.py`.

⛔ **I did not delete them** (COMMON §5.1 is absolute). ⭐ Instead I **copied the results into a staging tree
inside my own write prefix** and packed that, leaving every original byte in place on FORGIS_14:

| | |
|---|---|
| returned | **112 files, 5,552,257 B → `hub-generation-table.return.tar.gz` 3,494,205 B** |
| `sha256` | `9d18c3c51d23282feeb7df17b6c26941d63efa90d60fbbbb3831fe6f1a391d49` |
| left in place, **not** returned | **78 files, 2,176,458,180 B**, each listed with its size and sha256 in `RETURN-EXCLUSIONS.json`, with what it is and how to regenerate it |
| ingest | `scripts/bundle_hub_workspace.sh --return-ingest <tar>` — the returned tree has the required `hub-generation-table/` prefix and its own `RETURN_MANIFEST.json` |

⚠ **If the Head wants the full 2 GiB instead, it is untouched on the box** and one `--return-pack` away.

## ⏱ ELAPSED TRUE WALL-CLOCK, AND THE VERDICT
**START `2026-09-16T19:10:50Z` → `2026-09-17T02:32Z` = 7 h 21 m** of the 12 h window
(measured against `START_UTC.txt` and the real clock, ⛔ not against my sense of elapsed time).

> # ⛔ **NOT SOLVED — truthfully.**
> **0 of 3 scored runs spent**, blocked by COMMON §1c (budget never given) **and** by the pin's own
> re-encode gate, which fails on this machine on all three seeds and aborts `exp_gen_cell` before it
> computes any held-out number. ⛔ Neither was worked around.
> On 49 held-out DEV corners the arm **loses to `knn_proto` by −0.0172 (16/49)** ⇒ **falsifier 1 fails at the
> ceiling clause**, exactly as I predicted in writing before measuring it.
> ⭐ **Falsifier 2 passes** (0.8135 with **49.9 distinct tuples of 72** against a **0.5278** constant floor).
> ⭐ **Falsifier 3 passes on §C.2 everywhere**, and on §C.1 **only at `K = 32`**, ⛔ not at the shipped `K = 5`.
>
> ⭐ **What I would not trade for a win:** the free floor is **0.5278**, not 0.3272; `knn_proto` is **not a
> ceiling** but the best **additive** model, beaten by **+0.1285 (6/6)** by a pairwise one; and **the shipped
> read is one vertex of a span its own modes already form — changing only the read verb, at a byte-identical
> ledger, is worth +0.1192 (6/6) and takes distinct tuples from 9.5 to 35.0.**
> Three of my own claims were retracted along the way, each caught by building the harder control.

⭐ **Stopping here with an honest null and three retractions is the outcome the charter asks for. A
manufactured win would have been the one unrecoverable failure, and there were several moments in this run
where one was available.**

## ⛔⛔ 2026-09-17T02:31Z — the blocker is NOT specific to my module. **13 of 16 call sites enforce it.**

I audited every call site of `_assert_cache_reencodes` in the pinned tree
(`dev/reencode_call_sites.json`). ⛔ **Only ONE genuinely scopes the gate to the hazard**:

| call site | `will_extend` | aborts on this box? |
|---|---|---|
| `exp_gen_atom_mean.py:802` · `exp_gen_attribution.py:265` · `exp_gen_baselines.py:1508` · **`exp_gen_cell.py:1780`** · `exp_gen_codebook_alloc.py:337` · `exp_gen_codebook_exponent.py:290` · **`exp_gen_compose.py:408`** · `exp_gen_dream.py:519` · `exp_gen_dynamical_read.py:507` · `exp_gen_store.py:284` · `exp_gen_variance_eq.py:470` | ⛔ **DEFAULT `True` — ENFORCED** | ⛔ **yes** |
| `exp_gen_certificate.py:292` · `exp_gen_classifier.py:315` · `exp_gen_store.py:760` | `True` (explicit) | ⛔ **yes** |
| ⭐ **`exp_gen_wells_route_b.py:382`** | ⭐ `bool(cov["n_shortfall"] > 0)` | ✅ **no, when coverage is complete** |

> ⛔⛔ **13 of 16 call sites abort on FORGIS_14.** The pin ships the scoping mechanism and exactly **one**
> module uses it. ⇒ **this is not a `hub/generation-table` problem and not an `exp_gen_cell` problem — it is a
> property of the bundle on this machine, and it will stop the other gen experiment modules too.**

⚠ **Stated carefully:** I am reporting a **code fact** about the shared pinned tree plus a **measured gate
failure**. ⛔ I have not read any peer hub's outputs, and I do not know which entry points they use — whether
this bites them depends on what they run.
⭐ **The fix is one line per call site, and `exp_gen_wells_route_b.py:382` is the worked exemplar already in
the tree.** ⛔ Not mine to apply (charter §A.4).

## 2026-09-17T02:33Z — ⛔ STOPPING HERE, at 7 h 22 m of 12 h, and why that is the right call

I have **4 h 38 m** of the window left and I am stopping. The reason is not fatigue, and it is not that a
number looks presentable — it is that **more time cannot change the verdict**:

1. ⛔ **The thing that would settle the target — a scored run on `eval_b` — is unreachable from this machine.**
   Two independent hard blocks, both measured, neither mine to remove. **No amount of further DEV work can
   produce a number on `eval_b`.**
2. ⛔ **The DEV comparison that decides falsifier 1 is already well powered:** `n = 49` corners,
   `Δ = −0.0172 ± 0.0089`. Another hour moves the SE from 0.0089 to ~0.008. **It cannot flip the sign.**
3. ⛔ **Nine mechanism families are exhausted and each failure is explained, not merely observed:** vertex
   reads across K; launch-then-snap (`arm ≤ head_alone`, structurally); low-rank logit heads; low-rank of the
   whole head; decoder-diagonal-weighted codebooks; full-span (disqualified by §C.2 *before* running); all-15
   one-hot pair blocks; fully-smooth and hybrid request bases; and the arm itself. The byte arithmetic
   `head = n_cols · K·M` against `store = M·D` is a **closed trade-off**, and I have written down where it closes.
4. ⚠ **The host is out of commit space** — DEV-29 died on it at 02:28Z, and corner times had already tripled.
   Pushing further risks corrupting a run rather than adding to one.

⭐ **What I would rather leave behind than a tenth attempt at the ceiling:** a running log with every
measurement and **three of my own claims retracted** in it, a pre-registration whose prediction was right
about the conclusion, a ledger that says **0 scored runs** and why, and one recommendation — **change the read
verb** — that is worth **+0.119 (6/6)** at a byte-identical ledger and costs the programme nothing to try.

⛔ **`hub/generation-table`: NOT SOLVED. Reported as such.**
