# Task: gen-encode-precision — ⛔ the GPU does not reproduce the feature cache. Fix the PRECISION, ⛔ not the tolerance.

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-encode-precision.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · `chlu/core/gen_bridge.py:855` (`encode_images`) · `chlu/experiments/exp_gen_store.py` (`_assert_cache_reencodes`, the gate that caught this)
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Base:** the Hub's tip · **Branch:** `agent/experiment-engineer/gen-encode-precision` · **Worktree:** `../CHLU-prec`
- **Env:** main venv. pytest with `CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen`; report **passed / skipped / failed**.
- ⚠ **Budget ≈ 3 hours. LEAN.** ⛔ No PREREG — a correctness repair. ⛔ No sub-agents. ⛔ No number enters any table.

## ⭐ DIAL DECLARATION
- **Dial:** ⛔ **none — a correctness repair.** **Falsifies:** nothing.
- **Laundering control:** ⭐ **CPU results must be BIT-IDENTICAL before and after.** If the laptop's numbers move,
  the fix has changed the science and is wrong.

## The defect, measured on CSF3 (L40S, Ada sm_89, 2026-09-16)
The frozen encoder re-encoding its own cached rows **on GPU**:
```
max |delta| = 0.00390625 (2^-8) = 4181 ULP of float16
27332 / 131072 elements (20.85 %) differ · n_nonfinite = 0 · encoder param hash MATCHED
```
⭐⭐ **The gate was RIGHT to refuse. ⛔ This is not a rounding flip and ⛔ it must not be tolerated away.**
- The error is **bounded in ABSOLUTE terms** (≤ 2^-8, one float16 ULP at the tensor's *max* magnitude ~4–8) and
  **unbounded in RELATIVE terms** (4181 ULP on an element whose cached value is ~1e-3). ⇒ the error is set by the
  scale of the **partial sums**, ⛔ not by the outputs — the signature of **reduced-precision matmul**.
- ⭐ **`grep` finds NO matmul-precision setting anywhere in `chlu/`.** JAX's GPU default permits **TF32**
  (~10 mantissa bits, ≈1e-3 relative) for float32 convolutions on Ampere/Ada. **The cache was written on CPU in
  true float32.**
- ⚠ A peer hit the same family: `gen-cloud-enablement` needed forced XLA determinism or a resumed run agreed only
  to ~6 significant figures.
⇒ ⛔⛔ **THE STAKE IS NOT A FAILING PREFLIGHT.** If a GPU run *extends* the cache, it writes features inconsistent
with the ones already in it, and ⛔ **nothing downstream would notice.**

## The one job
**Make the frozen encoder produce the same features on GPU as it does on CPU.**

## Scope
1. ⭐ **Force full float32 precision on the frozen-encoder encode path** — `encode_images` in
   `chlu/core/gen_bridge.py`. Prefer an **explicit, local** precision (`jax.lax.Precision.HIGHEST` /
   `precision=` on the ops, or a scoped `jax.default_matmul_precision` context) over a global config mutation.
   ⛔ **Do NOT set a global flag at import time** — six modules already mutate JAX config at import and it is a
   standing hazard (§7.23).
2. ⭐⭐ **REPORT THE PRECISION ACTUALLY IN USE**, into the preflight record and the log: the requested precision, the
   backend, and the device kind. ⛔ A silent precision is how this hid in the first place.
3. **Leave `_assert_cache_reencodes` ALONE.** ⛔ Do not widen `ulp_bound`, ⛔ do not widen `frac_bound`,
   ⛔ do not add a GPU branch. ⭐ **It is the instrument that will tell us whether this worked.**
4. ⚠ **Say what it costs.** HIGHEST precision is slower than TF32. Measure the encode-path slowdown on CPU and
   **state the expected GPU cost**; if it is large, say so — ⛔ correctness first, but the number is owed.

## ⛔ You cannot test the thing that matters, and that is the point
⛔⛔ **CSF3 needs 2FA and there is no GPU on this laptop** (Apple Silicon, JAX runs CPU). ⇒ ⛔ **you cannot confirm
the fix.** ⭐ **The cluster run IS the test, and you must make it a sharp one:**
- ⭐ **State the falsifiable prediction in your report, before it runs:** with precision forced, the re-encode delta
  drops to **≤ 1 ULP on a handful of elements** and the gate passes. ⛔ **If 20 % of elements still differ, TF32 was
  NOT the cause** — and then ⛔ **nobody widens the tolerance**; it comes back for a different diagnosis.
- ⭐ **Emit one copy-able submit line** for the Head, `-p gpuL`, carrying `GEN_GIT_COMMIT`.
- ⭐ **Report every cluster leg `NOT RUN`.**

## Files you own — and nothing else
- `chlu/core/gen_bridge.py` — ⛔ **`encode_images` and its helpers ONLY**
- `tests/test_gen_bridge.py` · `.claude/outputs/gen-encode-precision/**`
⛔ **Not `chlu/experiments/exp_gen_store.py`** (the gate — a peer just repaired it), ⛔ not `exp_gen_pretrain.py` /
`exp_gen_bridge.py` / `exp_gen_wells_route_b.py` (callers — the signature stays backward-compatible),
⛔ not `chlu/config.py`, ⛔ not `scripts/**`.

## Acceptance
1. ⭐⭐ **CPU BIT-IDENTITY, before vs after.** Encode a fixed batch on `main` and on your branch; **every byte equal**.
   ⛔ If CPU numbers move, the fix is wrong — the cache was written on CPU and must stay reproducible there.
2. Precision is explicit and **local**; ⛔ no import-time global mutation. Demonstrate the requested precision
   reaching the op (a test that fails if it is dropped).
3. The precision / backend / device kind appear in the preflight record **and** the log.
4. `_assert_cache_reencodes` **unchanged** — `git diff` on `exp_gen_store.py` is empty.
5. The falsifiable prediction stated **before** the run, plus the submit line. Cluster legs `NOT RUN`.
6. The cost measured on CPU and the GPU cost estimated. Targeted tests green. A **cost line**.

## What would prove this spoke wrong
- ⭐⭐ **You widened the tolerance instead.** That buries a real defect: a GPU cache-extension would then write
  inconsistent features and ⛔ nothing downstream would catch it.
- ⭐ **CPU numbers moved.** Then every cached feature and every asserted hash in the programme is now suspect.
- ⭐ **You set a global JAX flag at import**, joining the six modules that already do and making §7.23's ordering
  hazard worse.
- ⭐ **You claimed the fix works without a GPU.** You cannot. Predict, hand over the submit line, and say NOT RUN.
