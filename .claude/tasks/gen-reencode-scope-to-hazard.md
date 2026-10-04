# Task: gen-reencode-scope-to-hazard — ⭐ the gate refuses runs that cannot trigger the hazard it guards. Scope it.

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-reencode-scope-to-hazard.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · `chlu/experiments/exp_gen_store.py` (`_assert_cache_reencodes`) · `chlu/experiments/exp_gen_wells_route_b.py:356-377` (coverage gate, then re-encode gate) · `.claude/outputs/gen-encode-precision.md`
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Base:** the Hub's tip · **Branch:** `agent/experiment-engineer/gen-reencode-scope-to-hazard` · **Worktree:** `../CHLU-scope`
- **Env:** main venv. pytest with `CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen`.
- ⚠ **Budget ≈ 3 hours. LEAN.** ⛔ No PREREG. ⛔ No sub-agents. ⛔ No number enters any table.

## ⭐ DIAL DECLARATION
- **Dial:** ⛔ **none — a correctness repair.** **Falsifies:** nothing.
- **Laundering control:** ⭐ **the gate must still ABORT, at full strength, on any run that will extend the cache.**

## ⛔⛔ READ THIS FIRST — the Hub has been wrong about this gate TWICE
1. ⛔ *"The gate is over-strict, it fired on a rounding bit."* **WRONG** — 20.85 % of elements differed.
2. ⛔ *"Force float32 and it drops to ≤ 1 ULP."* **WRONG** — it dropped to **53 ULP on 0.964 %**, not ≤ 1.
⇒ ⛔⛔ **DO NOT WIDEN `ulp_bound`. DO NOT WIDEN `frac_bound`. DO NOT ADD A GPU BRANCH.** ⭐ **If your answer
involves a bigger number, it is the wrong answer and this task has failed.**

## What the precision fix DID achieve (measured on gpuL, then again on an L4)
| | before (TF32) | after (HIGHEST) |
|---|---|---|
| elements differing | 27,332 (**20.85 %**) | 1,264 (**0.964 %**) — 21.6× fewer |
| max ULP | 4,181 | **53** — 79× smaller |
⭐ **Two different Ada GPUs reported the SAME 20.85 % before the fix** ⇒ deterministic, ⛔ not noise.
⚠ **0.964 % sits just under `frac_bound = 1.0 %`** — it failed on the ULP bound. ⛔ That is not a comfortable margin.

## ⭐⭐ THE ACTUAL INSIGHT — what the gate protects, and what the run does
The gate exists because **a run that EXTENDS the cache on a GPU would write features inconsistent with the rows
already in it, and nothing downstream would notice.** ⭐ **That is a real hazard and it keeps its full teeth.**
⛔ **But route B's own preflight, two lines earlier, reports `cache coverage: 33864/33864 — 0 shortfall`.**
⇒ ⭐⭐ **THE RUN ENCODES NOTHING NEW. The gate is refusing on a capability the run never exercises.**

## The one job
**Make the re-encode gate binding exactly when the run will extend the cache, and informational when it cannot.**

## Scope
1. ⭐ **`_assert_cache_reencodes` gains an explicit `will_extend` (or equivalent) parameter.**
   ⛔⛔ **It DEFAULTS TO STRICT.** A caller that passes nothing gets today's behaviour — abort on mismatch.
   ⛔ **No silent loosening is acceptable**; the relaxation must be *asked for*, per call site, in the open.
2. ⭐ **Each of the four callers passes it from what it already knows.** `exp_gen_wells_route_b` has `cov` in hand
   (`n_covered` / `n_rows_needed`) immediately above the call. ⛔ Do not re-derive coverage — **use the gate's own
   answer**, so the two can never disagree.
3. ⭐ **When NOT enforcing, still report — loudly and in the manifest**: the five statistics, plus `enforced: false`
   **and the reason**. ⛔ A skipped check that leaves no trace is how this comes back as a mystery.
4. ⭐⭐ **THE HAZARD MUST STILL BE CAUGHT AT THE POINT OF EXTENSION.** If a run passes an unenforced preflight and
   then encodes new rows anyway, **that path asserts at full strength**. ⛔ Preflight is not the only place the
   hazard lives, and scoping preflight must not open a hole behind it. ⚠ If the extension path is genuinely
   unreachable from a covered run, **prove it and say so** rather than adding a check that cannot fire.
5. ⚠ **ONE PASS, capped, on the residual.** `gen-encode-precision` forced `HIGHEST` on two conv equations.
   **Look once** for any other op on the encode path still on a default precision (a linear layer, a
   normalisation, an einsum). ⭐ **Report what you find. ⛔ You are NOT required to reach ≤ 1 ULP** — CPU and GPU
   convolutions use different reduction orders and exact agreement may be unattainable. ⛔ Do not spend the budget
   here; item 1–4 is the deliverable.

## Files you own — and nothing else
- `chlu/experiments/exp_gen_store.py` (the function **and** its own call site) · `chlu/experiments/exp_gen_classifier.py`
  · `chlu/experiments/exp_gen_certificate.py` · `chlu/experiments/exp_gen_wells_route_b.py` — ⛔ **the
  `_assert_cache_reencodes` CALL SITES ONLY** in the latter three
- `chlu/core/gen_bridge.py` — ⛔ **only if item 5 finds a residual op**, and then the encode path only
- `tests/test_gen_store.py` · `.claude/outputs/gen-reencode-scope-to-hazard/**`
⛔ Not `chlu/config.py`, ⛔ not `scripts/**`, ⛔ not any other module.

## Acceptance
1. ⭐⭐ **THE TEETH, DEMONSTRATED:** a run that **will extend** the cache and whose encoder disagrees **still
   ABORTS**, at today's bounds. ⛔ This is the whole point; if it cannot be shown failing, nothing ships.
2. ⭐ **A fully-covered run PASSES** with the disagreement **reported** — `enforced: false`, the reason, and all
   five statistics in the returned dict and the log.
3. **Default is strict:** calling without the new parameter aborts exactly as today. Pin it with a test.
4. ⭐ **The extension path asserts at full strength** (item 4) — demonstrated, or its unreachability proven.
5. `ulp_bound` and `frac_bound` **unchanged** — `git diff` shows neither constant edited.
6. Item 5 reported: residual op found (and fixed) **or** a stated "none found, and here is where I looked".
7. Targeted tests green, **passed / skipped / failed**. ⛔ No full suite. A **cost line**.

## What would prove this spoke wrong
- ⭐⭐ **You widened a bound.** The Hub has twice mistaken this gate for the problem; ⛔ do not make it a third time.
- ⭐⭐ **A run that extends the cache now slips through**, because the flag defaulted permissive or a caller passed
  the wrong thing. That writes inconsistent features and ⛔ **nothing downstream would ever catch it.**
- ⭐ **You re-derived coverage** instead of using the coverage gate's own answer, so the two can disagree.
- ⭐ **The unenforced path leaves no record**, so a future reader cannot tell a skipped check from a passed one.
