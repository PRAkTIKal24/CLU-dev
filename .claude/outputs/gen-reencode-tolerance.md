# gen-reencode-tolerance — experiment-engineer report

Task + acceptance criterion: make `_assert_cache_reencodes` correct across hardware without losing its
teeth — tolerance derived from the cache dtype, positive control still convicts a wrong encoder.
Status: **done**

## ⛔ RECONCILIATION LIST — needs an owner (protocol §5 corollary; in the first 10 lines as required)
Three **doc/log** sites still say "bit-identical" and are now false. ⛔ I did **not** edit them — they are
outside the one function I own (two are inside caller modules). All three are one-liners:
1. `chlu/experiments/exp_gen_store.py:237` — `gather_features` docstring: *"re-encodes a sample of genuinely
   cached rows and requires **bit-identity**"* → "…requires agreement to within one ULP of the cache dtype".
2. `chlu/experiments/exp_gen_atom_mean.py:805` — log string `"re-encode check bit-identical on {n} rows"`.
3. `chlu/experiments/exp_gen_cell.py:1295` — the same log string.
⚠ Sites 2–3 also *discard* the new statistics; they could log `max_ulp_delta`/`frac_differing` instead.

## ⭐ DIAL DECLARATION (echoed)
- **Dial:** ⛔ **none — a correctness repair.** **Falsifies:** nothing. ⛔ No number here enters any table.
- **Laundering control:** ⭐ the **positive control** — the gate must still convict a genuinely wrong
  encoder. Executed and measured below (§Findings 1).

## What I did
- Rewrote **only** `_assert_cache_reencodes` in `chlu/experiments/exp_gen_store.py`. Every diff hunk in that
  file is inside that function (verified: `git diff -U0 | grep '^@@'` — all 8 hunks carry that function's
  header). ⛔ Zero caller edits, zero `chlu/config.py`, zero `chlu/core/**`.
- **Tolerance derived from the cache dtype.** Per element, the bound is `np.spacing` evaluated on **that
  element's own cached value cast to `cman["dtype"]`**. ⛔ No epsilon, no `atol`, no GPU flag, no single
  global ULP: the bound is 0.00098 near 1.0 and 0.0078 near 8.0, automatically, and follows any future
  `cman["dtype"]`.
  - *Correctness of the scaling choice:* `np.spacing(|v|)` is the gap **above** `|v|`; the gap below is
    never larger (it halves only at a binade boundary). So a genuine one-representable-step move always
    scores ≤ 1 ULP (never falsely convicted) and nothing larger than one step is ever excused.
- **Second bound — the differing fraction ≤ 1%** (see the derivation below), because every element being
  within 1 ULP is not enough: half the tensor moving is a different computation.
- **NaN/Inf convicted explicitly.** `nan > 0` is `False`, so without this a NaN sails through every
  comparison and the gate becomes unable to fail. Counted before the arithmetic.
- **The five statistics are computed, logged and returned ALWAYS, pass or fail**, so they reach the manifest
  rather than only an exception string.
- **Failure message reports what was measured, in ULP.** ⛔ The string "this is NOT the encoder" is gone —
  the parameter hash is asserted separately and had already matched; the reader concludes.
- 21 new tests in `tests/test_gen_store.py`.

### The declared bounds and their derivation
| bound | value | derivation |
|---|---|---|
| `max_ulp_delta` | **1.0** | one representable step of the cache's own format, per element, via `np.spacing` at the cache dtype |
| `frac_differing` | **1e-2** | a float32 accumulation reorder perturbs the output by relative ~1e-7; float16 has relative precision 2^-11 ≈ 4.9e-4; an element re-rounds only within ~1e-7 of a rounding midpoint ⇒ expected flip fraction ≈ 2·1e-7/4.9e-4 ≈ **4e-4**. The cluster incident measured **1/131 072 = 7.6e-6**. So 1e-2 sits ~25× above the derivation and ~1300× above the observation, and ~50× below the 100%-of-elements signature of a wrong encoder (measured below). |

⛔ Both bounds are **function-local constants, deliberately NOT keyword arguments** — there must be no
per-caller, per-device way to widen or disable them. They are returned in the dict (`ulp_bound`,
`frac_bound`) so the manifest records the bound that was actually applied.

## How I verified
Main venv (`/Users/user/Desktop/CHLU/.venv`, **JAX 0.9.0** — ⛔ no `uv sync`, no worktree venv), worktree
`../CHLU-reenc` off `main = 9bf6422`.

```
CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen \
PYTHONPATH=/Users/user/Desktop/CHLU-reenc /Users/user/Desktop/CHLU/.venv/bin/python \
  -m pytest tests/test_gen_store.py -q -p no:cacheprovider --no-cov
```
⇒ **52 passed / 0 skipped / 0 failed / 0 warnings**, 120.49 s. (31 pre-existing + 21 new.)
⛔ No full suite (per the task). `ruff check` on both files: **All checks passed**.

The measured numbers below come from `.claude/outputs/gen-reencode-tolerance/demo.py` → `demo.json`
(the same code paths the tests assert on, run with the log captured so the statistics are visible on
**failing** runs too).

## Findings/results — the acceptance items, measured

**⭐⭐ 1. THE POSITIVE CONTROL — the gate still convicts a wrong encoder.** A real (tiny) `eqx.Module`
encoder driven through the **real** `encode_images`; the cache is written by the true encoder, then the
encoder's parameters are perturbed by a multiplicative factor.

| encoder | `max_ulp_delta` | `n_differing` | `frac_differing` | verdict | convicted by |
|---|---|---|---|---|---|
| **true encoder** | **0.000** | 0/256 | 0.000% | **PASS** | — |
| ×1.01 (1e-2 rel.) | **19.000** | 256/256 | **100.000%** | **RAISED** | ULP bound |
| ×1.0001 (1e-4 rel.) | 1.000 | 40/256 | **15.625%** | **RAISED** | ⭐ **fraction bound only** |
| ×1.000001 (1e-6 rel.) | 1.000 | 1/256 | 0.391% | PASS | — (see §Risks) |

⭐ The 1e-4 row is the fraction bound earning its place: it is the **only** thing that catches a
perturbation at that scale, because no single element exceeds one ULP. ⛔ The true encoder passing at
exactly 0 ULP is what makes the conviction mean something.

**⭐ 2. The cluster case now PASSES, at exactly 1 ULP.** One element of a float16 cache in the [4, 8)
binade moved by `0.00390625`:
```
⭐ cache re-encode check: 8 rows, 1/512 elements differ (0.195%),
  max |delta| = 0.00390625 = 1.000 float16 ULP (bounds: 1 ULP, 1.0% of elements)
```
⇒ `max_ulp_delta = 1.0`, no raise. ⭐ **Magnitude scaling demonstrated, not asserted:** the *identical*
absolute delta `0.00390625` applied at a cached value of **1.0** scores **4.000 ULP** and is **RAISED**.
That is the single-global-ULP failure mode firing on purpose.

**⭐ 3. Half the tensor off by 1 ULP is REFUSED by the fraction bound.**
```
⭐ cache re-encode check: 8 rows, 256/512 elements differ (50.000%),
  max |delta| = 0.00390625 = 1.000 float16 ULP (bounds: 1 ULP, 1.0% of elements)
AssertionError: ... stayed within one float16 ULP (max 1.000 ULP) but 256/512 elements (50.000%)
moved, over the 1.000% bound: a rounding-boundary crossing touches a handful of elements, a
population this size is a different computation — {...}
```
⛔ Note `max_ulp_delta = 1.000` — the per-element bound is satisfied. Only the fraction bound convicts.

**4. Returned dict + caller compatibility.** Signature is **byte-identical**
(`dataset, config, encoder, cache_rows, features, cman, n=64, log=print`) and pinned by a test. The dict
**only gains keys**:
`n_checked` (⛔ still **ROWS** — `exp_gen_atom_mean.py:805` and `exp_gen_cell.py:1295` print it as a row
count) · `max_abs_delta` (kept) · **new:** `max_ulp_delta`, `n_differing`, `frac_differing`, `n_elements`,
`n_nonfinite`, `cache_dtype`, `ulp_bound`, `frac_bound`.
⚠ **The task says four callers; there are 14 import sites.** All 14 were imported and asserted to bind the
same object, and all 14 are exercised with the exact call form the sites use
(`(dataset, config, encoder, crows, feats, cman, log=log)`), returning `n_checked == rows`:
`exp_gen_store` (own call site, :538) · `exp_gen_classifier` · `exp_gen_certificate` ·
`exp_gen_wells_route_b` · `exp_gen_atom_mean` · `exp_gen_attribution` · `exp_gen_baselines` ·
`exp_gen_cell` · `exp_gen_codebook_alloc` · `exp_gen_codebook_exponent` · `exp_gen_compose` ·
`exp_gen_dream` · `exp_gen_dynamical_read` · `exp_gen_variance_eq`. ⛔ None edited.

**5. Cost line.** ~1 h 40 m wall (budget ≈ 2 h). Compute: CPU only — `pytest tests/test_gen_store.py`
120 s ×3 runs + a 15 s targeted run + ~25 s of demo script. ⛔ No GPU, no cluster, no sub-agents, no
`uv sync`. Module import ~15 s (⚠ not the ~20 min the protocol warns about, on this venv).

### The 21 new tests
`..._accepts_the_cluster_case_one_float16_ulp` · `..._tolerance_tracks_magnitude_not_a_global_ulp` ·
`..._convicts_a_perturbed_encoder_POSITIVE_CONTROL` · `..._refuses_half_the_tensor_off_by_one_ulp` ·
`..._refuses_a_nan` · `..._reports_the_five_fields_on_a_clean_pass` · `..._signature_is_unchanged` ·
`..._callers_still_work_unchanged[14 modules]`.

## Git footprint
- **Branch:** `agent/experiment-engineer/gen-reencode-tolerance` · **base:** `main = 9bf6422` (the Hub's
  tip; a clean fast-forward — rebase onto the named base is a no-op, `origin/main` never touched per §7.21).
- **Commit:** `9a8cab7` — *"[experiment-engineer] derive the cache re-encode tolerance from the cache dtype"*.
- **Files touched:** `chlu/experiments/exp_gen_store.py` (+110 / −11, all inside `_assert_cache_reencodes`) ·
  `tests/test_gen_store.py` (+253 / −0).
- **Worktree:** `../CHLU-reenc`, created off `main`, **branch ref verified from the main repo before
  removal** (`git log --oneline main..agent/…` showed `9a8cab7`), then `git worktree remove`d. Worktree
  count is back to its pre-task value; ⛔ the shared checkout at `/Users/user/Desktop/CHLU` (on
  `pilot-ttt-nan-and-d5-wiring`) was never edited. No conflicts.
- Scratch/artifacts: `.claude/outputs/gen-reencode-tolerance/{demo.py,demo.json}`.

## Open questions / follow-ups / risks
1. ⚠ **Detection floor (honest limitation).** A **1e-6 relative** parameter perturbation PASSES (1 element,
   1 ULP). That is not a loosening — at that scale the perturbation is genuinely below the cache's own
   quantisation and is indistinguishable from the float-reorder it must tolerate. ⛔ **The parameter-hash
   assertion, not this gate, is the defence against a swapped encoder**; this gate defends against a hash
   that matches but a computation that does not. Stated so nobody later reads the gate as stronger than it is.
2. ⚠ **Exact-zero cached elements use the float16 *subnormal* ULP (5.96e-8).** If the encoder emits a value
   the CPU rounded to `0.0` and the GPU rounds to ≥2 subnormal steps (≥1.2e-7), that element scores >1 ULP
   and the gate fires. ⛔ **Not observed** — the cluster incident was a single element in [4, 8) — and I had
   no encoder checkpoint here to measure the exact-zero population on a real Shapes3D cache. ⭐ **If the
   gate fires again on the cluster, check `max_abs_delta` first: a value of order 1e-7 is this case, not a
   wrong encoder.** Deliberately not floored with an epsilon, which is exactly what the task forbids.
3. ⚠ **The fraction bound needs a large sample to be meaningful.** At production `n=64` rows × `D=2048`,
   `n_elements = 131 072` and 1% = 1310 elements — fine. At a hypothetical `n_elements < 100`, a *single*
   differing element would exceed 1%. ⛔ **No caller passes `n`** (all 14 verified), so this is not live;
   flagged so a future caller does not set `n` small and get a false conviction.
4. The three doc/log sites in the reconciliation list at the top of this report.

## Proposed handover updates (for the Hub)
- **§7 Known Issues — new entry, RESOLVED.** *"7.30 [RESOLVED 2026-09-15 by `gen-reencode-tolerance`,
  `9a8cab7`] `_assert_cache_reencodes` demanded bit-identity and blocked every gen stage on a GPU node.*
  *Cluster job 20603452 (L40S): one element of 64×2048 moved by `0.00390625` = exactly 2^-8 = **one float16
  ULP in [4, 8)**, and the gate convicted an encoder whose parameter hash had matched in the same preflight.
  A GPU accumulates the encoder's convolutions in a different order than the CPU that wrote the float16
  cache, so bit-identity across hardware is **unachievable and is not what 'the same encoder' means**. The
  gate now requires agreement to within **one ULP of the cache dtype, scaled per element by `np.spacing` on
  its own cached value**, AND a differing fraction ≤ 1%. Teeth verified: a 1%-perturbed encoder is convicted
  at **19.0 ULP / 100% of elements**; a 1e-4-perturbed one is caught by the **fraction bound alone** (1.0
  ULP, 15.6%). ⛔ **The gate has a detection floor**: a 1e-6 relative perturbation passes — the parameter
  hash, not this gate, is the defence against a swapped encoder. ⛔ **All 14 import sites, not four**, were
  verified unchanged."*
- **Standing note for the cluster:** the gate's log line is now emitted on every gen preflight, pass or
  fail, and carries `max_ulp_delta` + `frac_differing`. If it fires, the **ULP figure and the fraction**
  are the diagnosis; `max_abs_delta ~ 1e-7` means the exact-zero case in risk 2, not a wrong encoder.
- **Curator/engineer task to file** (the reconciliation list above): three "bit-identical" doc/log strings,
  one of them (`exp_gen_store.py:237`) in a file a future spoke will open.
