# Task: gen-reencode-tolerance — ⛔ a correctness gate is firing on a ROUNDING BIT and blocking every GPU gen run.

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-reencode-tolerance.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · `chlu/experiments/exp_gen_store.py:271` (`_assert_cache_reencodes`)
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Base:** the Hub's tip · **Branch:** `agent/experiment-engineer/gen-reencode-tolerance` · **Worktree:** `../CHLU-reenc`
- **Env:** main venv. pytest with `CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen`; report **passed / skipped / failed**.
- ⚠ **Budget ≈ 2 hours. LEAN.** ⛔ No PREREG — this is a correctness repair, not a measurement. ⛔ No sub-agents.
- ⛔ **No number from this spoke enters any table.**

## ⭐ DIAL DECLARATION
- **Dial:** ⛔ **none — a correctness repair.** **Falsifies:** nothing.
- **Laundering control:** ⭐ **the positive control below — the gate must still convict a genuinely wrong encoder.**

## The defect, measured on the cluster 2026-09-15 (job 20603452, L40S)
`_assert_cache_reencodes` demands **exact bit-identity** (`if max_abs != 0.0: raise`). On an L40S it failed:
```
AssertionError: re-encoding 64 cached rows through the loaded encoder did not reproduce
the cache (max |delta| = 0.00390625) — this is NOT the encoder that wrote the cache
```
⭐⭐ **`0.00390625 = 2^-8` is EXACTLY ONE float16 ULP for values in [4, 8)** — and the cache is **float16**
(`.astype(cman["dtype"]).astype(np.float32)`). ⭐ **The encoder hash `98d32602…` had already matched, in the same
preflight, moments earlier.**
⇒ ⛔ **The encoder is CORRECT.** One element of 64 × 2048 = 131,072 crossed a float16 rounding boundary because a GPU
accumulates convolutions in a different order than the CPU that wrote the cache. ⛔ **The message is a FALSE
ACCUSATION** and the gate is unachievable across hardware.
⛔⛔ **It blocks FOUR callers** — `exp_gen_store`, `exp_gen_classifier`, `exp_gen_certificate`, and
`exp_gen_wells_route_b`'s preflight ⇒ **every gen stage on a GPU node.** ⭐ It fails fast, so no compute was wasted.

## The one job
**Make the re-encode check correct across hardware without losing its teeth.**

## ⛔ How, and how NOT
⛔⛔ **NOT a hand-picked epsilon.** ⛔ NOT `atol=1e-3`. ⛔ NOT a flag that switches the check off on GPU.
⭐ **The tolerance is DERIVED FROM THE CACHE DTYPE: agreement to within ONE ULP of the format the cache is STORED in
IS the same encoder.** Scale per element by magnitude (`np.spacing` on the cached value at the cache dtype) —
⛔ a single global ULP is wrong, because ULP depends on magnitude (0.00098 near 1.0, 0.0078 near 8.0).
⭐ **Report ALWAYS, pass or fail** — and put them in the returned dict so they reach the manifest:
`max_abs_delta` · `max_ulp_delta` (in units of cache-dtype ULP) · `n_differing` · `frac_differing` · `n_checked`.
⚠ ⛔ **A run where MOST elements differ by 1 ULP is suspicious even if none exceeds it** — ⭐ **also require the
differing FRACTION to be small** (declare the bound and justify it). One bit on one element is rounding; half the
tensor moving is not.
⭐ **Keep the failure message accurate:** on a real mismatch say the measured delta **in ULP**, ⛔ not "this is NOT
the encoder" when the hash matched — say what was measured and let the reader conclude.

## Files you own — and nothing else
- `chlu/experiments/exp_gen_store.py` — ⛔ **the function `_assert_cache_reencodes` ONLY**
- `tests/test_gen_store.py` · `.claude/outputs/gen-reencode-tolerance/**`
⛔ **Do NOT touch the four callers** (`exp_gen_classifier.py`, `exp_gen_certificate.py`,
`exp_gen_wells_route_b.py`, and `exp_gen_store`'s own call site) — the signature and the returned dict stay
**backward-compatible**; they only gain keys. ⛔ Not `chlu/config.py`, ⛔ not `chlu/core/**`.

## Acceptance
1. ⭐⭐ **THE POSITIVE CONTROL — this is the whole acceptance.** The gate **still CONVICTS a genuinely wrong
   encoder**: perturb the encoder's parameters and show it raises, reporting a delta of **many ULP**.
   ⛔ **A gate that cannot fail proves nothing, and loosening one is how a real defect gets through.**
2. ⭐ **The cluster case now PASSES:** a cached tensor and a copy with **one element moved by exactly one float16
   ULP** (`0.00390625` at a value in [4,8)) is accepted, and the report shows `max_ulp_delta = 1`.
3. ⭐ **A half-the-tensor-off-by-1-ULP case is REFUSED** by the fraction bound. ⛔ Demonstrate it firing.
4. The returned dict carries the five fields above; **all four callers still work unchanged** — import them and
   exercise the call sites.
5. Targeted tests green with the data root set, **passed / skipped / failed**. ⛔ No full suite. A **cost line**.

## What would prove this spoke wrong
- ⭐⭐ **You made the gate unable to fail.** That is worse than the bug — this check is the only thing standing
  between a swapped encoder and a silently invalid run.
- ⭐ **You picked a round-number epsilon** instead of deriving the tolerance from the cache dtype, so it is arbitrary
  and will be wrong for the next dtype.
- ⭐ **You used a single global ULP**, so the tolerance is too tight near 1.0 and too loose near 8.0.
- You edited a caller, so the four sites drift apart.
