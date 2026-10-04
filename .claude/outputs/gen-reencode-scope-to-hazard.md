# gen-reencode-scope-to-hazard — experiment-engineer report

Task + acceptance criterion: make the cache re-encode gate **binding exactly when the run will extend the cache,
and informational when it cannot** — ⛔ without widening `ulp_bound` or `frac_bound`, and without opening a hole
behind the preflight.
Status: **done.** 7/7 acceptance criteria met. **111 passed / 0 skipped / 0 failed**, ⛔ no full suite.

⛔ **No reconciliation list — nothing here needs a downstream owner.** (§6 closes the one cross-report question I
opened, with no action required.)

## ⭐ DIAL DECLARATION (echoed)
- **Dial:** ⛔ **none — a correctness repair.** **Falsifies:** nothing.
- **Laundering control:** ⭐ **the gate must still ABORT, at full strength, on any run that will extend the cache.**
  ✅ **HELD, and shown FAILING** — §3(A) and §3(C), two independent tests.
- **Does NOT falsify:** a covered run passing with a *reported* disagreement is the intended behaviour, not a
  loosened gate. The bounds are unchanged and are asserted unchanged by a test.

---

## 1. What I changed

`_assert_cache_reencodes` gained **two keyword-only parameters**: `will_extend: bool = True` and `reason: str = ""`.

⛔⛔ **Neither is a tolerance, and I widened nothing.** `ULP_BOUND = 1.0` and `FRAC_BOUND = 1e-2` are byte-untouched,
still local to the function body, still not reachable from any caller. `will_extend` decides **whether the bounds
bind**, never how wide they are. `git diff main..HEAD | grep -E '^[+-].*(ULP_BOUND|FRAC_BOUND) *= '` returns
**exactly one line, and it is a test asserting the constants are still there**.

| | behaviour |
|---|---|
| `will_extend=True` (**the default**) | today's abort, exactly. The **10 caller modules I did not touch** are unaffected. |
| `will_extend=False` | statistics still computed, still logged; ⛔ a `reason` is **mandatory** (`ValueError`, not a quiet skip); the returned dict carries `enforced`, `enforced_reason`, `would_have_failed`, `would_have_failed_why`. |

And — because scoping a *preflight* must not open a hole behind it — **`gather_features`, the one function in the
programme that extends the cache, now runs the same check at the point of extension, ENFORCED, with no parameter
that can ask for less.**

## 2. ⭐⭐ The insight the task states, made mechanical

The hazard is a run that **writes new rows** into the cache's feature space through an encoder that does not
reproduce the rows already there. A run that encodes nothing new cannot trigger it. So the verdict is now taken
from **the coverage gate's own answer**, three lines above the call in `exp_gen_wells_route_b.preflight`:

```python
will_extend=bool(cov["n_shortfall"] > 0),
reason=(f"cache-coverage gate: {cov['n_covered']}/{cov['n_rows_needed']} rows cached, "
        f"{cov['n_shortfall']} shortfall (seed {seed}, scope={cov['cache_scope']!r}) " + …)
```

⛔ **Coverage is never re-derived.** `assert_cache_coverage(` appears **once** in that module and `searchsorted`
appears **zero** times in `preflight` — both pinned by `test_route_b_uses_the_COVERAGE_GATES_OWN_answer`. The two
gates cannot disagree about the same run.

The other three callers **have no coverage gate to relax on**, and all three call `gather_features`, so they say
`will_extend=True` **explicitly, at the call site, with a reason** — the relaxation is asked for per site, in the
open, and its absence is visible too.

| caller | verdict | from what |
|---|---|---|
| `exp_gen_wells_route_b.preflight` | `cov["n_shortfall"] > 0` | ⭐ the coverage gate's own dict |
| `exp_gen_store.run_seed` | `True` | no coverage gate; `gather()` → `gather_features` |
| `exp_gen_classifier.run_seed` | `True` | no coverage gate; `gather_features` at leg 2 |
| `exp_gen_certificate.run_seed` | `True` | no coverage gate; `gather_features` called twice |
| the other **10** modules | `True` (default) | ⛔ untouched — they pass nothing and keep today's abort |

## 3. ⭐⭐ How I verified — the three behaviours, as they actually print

`.claude/outputs/gen-reencode-scope-to-hazard/gate_scoping_demo.log` (full text). Encoder disagreement is
**exactly 8 float16 ULP on 100.000 % of elements** — ⛔ 8× the ULP bound and 100× the fraction bound, i.e. a
disagreement neither bound can excuse.

**(A) ENFORCED — the run will extend, the encoder disagrees ⇒ ⛔ ABORT (acceptance 1, the teeth)**
```
  ⭐ cache re-encode check: 8 rows, 512/512 elements differ (100.000%), max |delta| = 0.03125 = 8.000 float16 ULP (bounds: 1 ULP, 1.0% of elements)
  ⭐ ENFORCED (enforced=true): this run will extend the cache
  ⛔ ABORTED: re-encoding 8 cached rows did not reproduce the cache to within one float16 ULP … (bound 1); 512/512 element…
```

**(B) NOT ENFORCED — the SAME disagreement, on a fully covered run ⇒ PASSES, on the record (acceptance 2)**
```
  ⭐ cache re-encode check: 8 rows, 512/512 elements differ (100.000%), max |delta| = 0.03125 = 8.000 float16 ULP (bounds: 1 ULP, 1.0% of elements)
  ⚠ NOT ENFORCED (enforced=false): cache-coverage gate: 33864/33864 rows cached, 0 shortfall (seed 0, scope='full') ⇒ this run encodes nothing new and cannot extend the cache
  ⛔⛔ THE RE-ENCODE CHECK WOULD HAVE ABORTED THIS RUN (ulp, fraction) and was NOT enforced, because: … ⭐ The run encodes nothing new, so no inconsistent feature can be written — but the encoder does NOT reproduce this cache on this machine, and any run that extends the cache here WILL abort.
      re-encoding 8 cached rows did not reproduce the cache … — {…full stats…}
  returned: {'max_abs_delta': 0.03125, 'max_ulp_delta': 8.0, 'n_differing': 512, 'frac_differing': 1.0,
             'n_checked': 8, 'ulp_bound': 1.0, 'frac_bound': 0.01, 'enforced': False,
             'would_have_failed': True, 'would_have_failed_why': ['ulp', 'fraction']}
```
⭐ All five statistics, `enforced: false`, the reason, **and the verdict the bounds would have returned** — in the
returned dict (hence the manifest) and in the log. ⛔ A skipped check cannot be mistaken for a passed one, and the
inverse case prints `⭐ (the check was not enforced, and it would have PASSED anyway)`.

**(C) ⭐⭐ THE HOLE, SHUT — (B) passed, then the run asks for an uncached row (acceptance 4)**
```
  _REENCODE_VERIFIED after the unenforced pass: set()
    ⭐ cache extension requested — re-encode check runs ENFORCED here
  ⭐ cache re-encode check: 8 rows, 512/512 elements differ (100.000%), max |delta| = 0.03125 = 8.000 float16 ULP …
  ⭐ ENFORCED (enforced=true): gather_features is about to EXTEND the cache (point of extension)
  ⛔ ABORTED AT THE POINT OF EXTENSION: re-encoding 8 cached rows did not reproduce the cache …
```
⭐ **The unenforced pass bought it nothing.** An unenforced check deliberately does **not** register in
`_REENCODE_VERIFIED` (printed `set()` above); an **enforced** pass does, so a strict run is not charged twice.
⛔ I did **not** take the "prove it unreachable" option — the extension path is reachable and it is demonstrated
aborting, which is strictly stronger than a proof.

## 4. Targeted tests — **111 passed / 0 skipped / 0 failed** (131.79 s)

`tests/test_gen_store.py` + `tests/test_gen_wells.py`, `-q -p no:randomly --no-cov -rs`. ⛔ No full suite (task).
Log: `.claude/outputs/gen-reencode-scope-to-hazard/pytest_targeted.log`. `ruff check` clean on all 5 touched files.

**13 tests added** (one parametrized over 3 modules), of which the two load-bearing ones must fail on purpose:
`test_scoped_gate_STILL_ABORTS_when_the_run_will_extend` (acceptance 1) and
`test_an_UNENFORCED_preflight_does_not_excuse_a_later_extension` (acceptance 4).
Also: the covered run passes with the full record · the default is strict when the caller says nothing
(acceptance 3) · `will_extend=False` without a reason is a `ValueError` · **no signature parameter can reach either
bound** (`{ulp_bound, frac_bound, tol, atol, rtol, bound} ∩ params = ∅`) · a fully-cached gather never encodes at
all · an agreeing encoder still extends · an enforced preflight discharges the extension check once · source pins
on all four call sites.

**3 tests updated:** the exact-stats-dict test (4 new keys), the signature test (now also pins that both gains are
**keyword-only** and that `will_extend`'s default **is `True`**), and the 14-caller test (now also asserts a caller
that says nothing gets `enforced: True`).

⚠ **One honest note on my own test arithmetic.** The first run had **3 failures, all mine, none in the
implementation**: I asserted an 8-ULP perturbation and measured 9. On the existing `[4, 8)` fixture, adding
`0.03125` to a cached value just under 8.0 crosses the binade, where float16's step doubles to `0.0078125` and
round-half-to-even lands the result `0.03515625` away — **9 ULP of the cached value, not 8**. `_disagreeing_fixture`
now draws from `[4.0, 7.5)`, clear of the boundary, so the perturbation is exactly 8 ULP at every element and the
tests assert an exact number rather than an inequality.

## 5. ⚠ Cost line

| | |
|---|---|
| **covered run (route B, the case this task is about)** | ⭐ **zero extra compute.** The preflight re-encode ran before and runs now — same 64 rows, same arithmetic. Two extra log lines, four extra dict keys. |
| **extending run** | ⭐ **zero extra compute in practice.** Route B's `cov` says `n_shortfall > 0` ⇒ the preflight is ENFORCED ⇒ it registers in `_REENCODE_VERIFIED` ⇒ `gather_features` skips its own check. |
| **worst case (unenforced preflight, then an extension anyway)** | **one** extra 64-row re-encode per (cache, encoder) per process — 64 images ≈ 2.89 GFLOP (`gen-encode-precision` §4), against a path that is already streaming the missing rows. Not once per `gather` call: memoised. |
| targeted test run | 111 tests, **131.79 s** wall, main venv, CPU |

## 6. ⭐ Item 5 — one capped pass on the residual: **none found, and here is where I looked**

Probe: `.claude/outputs/gen-reencode-scope-to-hazard/encode_path_precision_audit.py` (+ `.json`). It captures the
**shipped** callable `encode_images` hands to `eqx.filter_jit` — not a reconstruction — takes its jaxpr at the
**shipped default `GenBridgeConfig`** (`enc_channels=[32,64,128,128]`, `k=4`, `s=2`, `groups=8`, `(3,64,64) →
(128,4,4)`) and **walks nested jaxprs**, classifying all 184 equations.

- **Ops carrying a `precision` parameter: 4 × `conv_general_dilated`, every one at `(HIGHEST, HIGHEST)`.**
- **Ops on a DEFAULT precision: `[]` — empty.** ⛔ No linear layer, no `dot_general`, no einsum anywhere on the
  encode path. `GenEncoder` is `[Conv2d → GroupNorm → relu] × 4` and nothing else.
- **Residual, named and unfixable by this lever:** 12 × `reduce_sum` (+ `rsqrt`, `div`, `square`) — **GroupNorm's
  mean and variance**. `jax.default_matmul_precision` is a parameter of dot/conv primitives only and provably
  cannot reach a reduction. ⛔ This is not a new finding: it is suspect **(b)** in `gen-encode-precision` §5 and its
  §10.2, and it remains the named reason the GPU delta may land just over 1 ULP rather than far under it.
- ⛔ **I changed nothing in `chlu/core/gen_bridge.py`** — the file is byte-untouched, as item 5 permits only if a
  residual is found. None was.

⭐ **Cross-report check, closed, no owner needed.** `gen-encode-precision`'s table says "**2** conv equations"; I
count **4**. Not a discrepancy: its `jaxpr_precision_probe.py` builds a **2-block toy** encoder
(`GenEncoder((3,16,16), [4,8], …)`), so 2 was right for that probe. Mine is the first audit on the **shipped
4-block** encoder, and all 4 convs carry HIGHEST. ⛔ Nothing to reconcile.

## 7. Acceptance, line by line

| # | criterion | |
|---|---|---|
| 1 | extending run + disagreeing encoder still ABORTS at today's bounds | ✅ §3(A), §3(C); 2 tests |
| 2 | covered run PASSES with `enforced: false`, the reason, all five statistics in dict **and** log | ✅ §3(B) |
| 3 | default is strict; pinned by a test | ✅ `test_scoped_gate_is_STRICT_when_the_caller_says_nothing` + the signature test pins `default is True` |
| 4 | extension path asserts at full strength | ✅ **demonstrated**, not proven-unreachable — §3(C) |
| 5 | `ulp_bound` / `frac_bound` unchanged in `git diff` | ✅ §1; plus `test_the_bounds_are_not_reachable_from_any_caller` |
| 6 | item 5 reported | ✅ §6 — none found, with the search enumerated |
| 7 | targeted tests green, passed/skipped/failed, a cost line | ✅ **111 / 0 / 0**, §4–§5 |

## 8. Flag provenance

⛔ **No number in this report is a research result** and none enters any table — every figure is a property of a
stubbed fixture chosen to exceed the gate's bounds, or a wall-time.

| | |
|---|---|
| base | local `main @ 8e17bfc` (⛔ **not** `origin/main`, still frozen at `40c2f31` — §7.21) |
| branch commits | `5e8771a` (code), `1be697e` (tests) |
| seeds | `np.random.default_rng(0)` (demo + `_disagreeing_fixture`); existing fixtures unchanged |
| non-default flags | none. `ULP_BOUND = 1.0`, `FRAC_BOUND = 1e-2`, `ENCODE_PRECISION = "highest"` — all as `main` left them |
| env | main venv `/Users/user/Desktop/CHLU/.venv` (⛔ no `uv sync`), **JAX 0.9.0**, backend `cpu`, macOS/Apple Silicon, `CHLU_GEN_DATA_ROOT=.claude/data/gen`, `PYTHONDONTWRITEBYTECODE=1` |
| ⛔ not run | **any GPU leg.** The L40S/Ada behaviour is unchanged by this spoke and untested by it. |

## 9. Git footprint

- **Branch** `agent/experiment-engineer/gen-reencode-scope-to-hazard`, worktree `../CHLU-scope`, base local `main @ 8e17bfc`.
  `main` is an ancestor of the branch ⇒ rebase-onto-`main` is a no-op; none performed, no conflicts.
- **Commits:** `5e8771a` *"bind the re-encode gate to the hazard, not to every run"* · `1be697e` *"tests: the teeth,
  the scoped pass, and the extension path"*. **+489 / −19** over 5 files.
- **Files touched** (`git diff --numstat main..HEAD`): `exp_gen_store.py` **+133/−9** (the gate, `gather_features`'
  extension check, `_cache_identity` / `_REENCODE_VERIFIED`, its own call site) · `exp_gen_classifier.py` **+7/−1**
  and `exp_gen_certificate.py` **+7/−1** (⛔ **call site only**) · `exp_gen_wells_route_b.py` **+17/−2** (⛔ **call
  site only**) · `tests/test_gen_store.py` **+325/−6**.
- ⛔ **Not touched:** `chlu/core/gen_bridge.py` (byte-identical to `main`), `chlu/config.py`, `scripts/**`, the
  other 10 caller modules, any other test file.
- No pushes, no PR. Worktree `../CHLU-scope` left in place for Hub review; branch ref **verified visible from the
  main checkout** (`git -C /Users/user/Desktop/CHLU log --oneline main..agent/…` → both hashes).
- No collision: the shared checkout was on another agent's branch (`pilot-ttt-nan-and-d5-wiring`, clean) and I
  never edited it; `git status` in `../CHLU-scope` is clean.

## 10. Open questions / risks

1. ⚠ **A judgement call I made and want visible: the NaN conviction is scoped too.** When `will_extend=False`, a
   `n_nonfinite > 0` result is reported (`would_have_failed_why == ['nonfinite']`, ⛔⛔ log line) but does not
   abort — consistently with the other two bounds. Rationale: a NaN in the *re-encode* cannot hurt a run that
   never encodes, and a NaN in the *cache* is the cache checksum's business, not a 64-row sample's. ⛔ If the Hub
   disagrees, the change is one `if` and no bound moves — say so and I will make `nonfinite` unconditional.
2. ⚠ **`_REENCODE_VERIFIED` is process-scoped module state.** Deliberate (it memoises a deterministic check) but it
   means a long-lived process that swaps the encoder *under an unchanged manifest* would reuse a stale pass. The
   key includes `encoder_param_hash`, so the only way in is a manifest that lies about its own encoder — which is
   asserted separately by every caller before this point.
3. ⚠ **Three of the four callers can never relax**, because only route B has a coverage gate. That is the correct
   conservative answer today, not a finished state: if `exp_gen_store` / `classifier` / `certificate` ever gain an
   `assert_cache_coverage` call, they can pass its `cov` the way route B does. ⛔ Adding that gate was outside this
   task's file scope (call sites only).
4. ⛔ **Nothing here is confirmed on a GPU.** This spoke changes *when* the gate binds; it does not change what the
   L40S measures. `gen-encode-precision`'s prediction (§5 of that report) is still the open question and is still
   unaffected by anything I did.

## Proposed handover updates (for the Hub)

**§7 Known Issues — add (and it closes the practical half of the entry `gen-encode-precision` proposed):**
> **§7.2x — the re-encode gate was refusing runs that cannot trigger its hazard (fixed 2026-09-16, branch
> `agent/experiment-engineer/gen-reencode-scope-to-hazard` @ `1be697e`).** `_assert_cache_reencodes` guards one
> thing: a run that **extends** the feature cache through an encoder that does not reproduce the rows already in
> it writes inconsistent features and ⛔ nothing downstream notices. Route B's preflight was refusing on that
> capability while its own coverage gate, three lines above, reported `33864/33864 — 0 shortfall` — **the run
> encodes nothing new.** The gate now takes `will_extend` (⛔ **keyword-only, default `True` = today's abort**; the
> 10 untouched callers are unaffected) and a mandatory `reason` when relaxed. ⛔⛔ **`ULP_BOUND = 1.0` and
> `FRAC_BOUND = 1e-2` are byte-unchanged and unreachable from any caller — the bounds were never the problem and
> were not touched for a third time.** Route B derives the verdict from `cov["n_shortfall"]`, the coverage gate's
> own answer, so the two can never disagree. An unenforced check still computes and logs all five statistics and
> returns `enforced` / `enforced_reason` / `would_have_failed` / `would_have_failed_why`, so a skipped check is
> never mistakable for a passed one. ⭐ **And the hazard is now caught where it actually lives:**
> `gather_features` runs the same check at the point of extension, ENFORCED — an unenforced preflight
> deliberately does not discharge it. Both the abort and the covered pass are demonstrated
> (`.claude/outputs/gen-reencode-scope-to-hazard/gate_scoping_demo.log`); 111 targeted tests green.

**One decision for the Hub (⚠ low stakes, one `if`):** §10.1 — should the **NaN** conviction remain scoped with
the other two, or fire unconditionally even on a run that cannot extend the cache?

**And a note for whoever reads `gen-encode-precision` next:** its §7 reconciliation item (fold `encode_precision`
into the gate's returned `stats`) is ⛔ **still open and still unowned** — I did not do it; it was not in this
task's scope, and the gate's `stats` dict now has four more keys, so whoever lands it should add the fifth there
rather than at 14 call sites.
