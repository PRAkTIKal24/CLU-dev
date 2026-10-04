# Task: gen-addressed-read — ⭐ pre-register the LAUNCH-ONLY read and score it HONESTLY at matched bytes.

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-addressed-read.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · `.claude/outputs/gen-read-cell-arm.md` (the diagnostic this promotes) · `.claude/autonomous-hub-charter.md` §C (the floor)
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Base:** ⭐ **the PIN `ae907466ba2b5fbec124e8995cf4d7c32a1ab30e`** · **Branch:** `agent/experiment-engineer/gen-addressed-read` · **Worktree:** `../CHLU-addrread`
- **Env:** main venv. pytest with `CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen`.
- ⚠ **Budget ≈ 1 day.** ⛔ No sub-agents. ⛔ **This is a BOUNDED SPOKE, not a hub.** ⛔ No ladder, no iteration.

## ⭐ DIAL DECLARATION
- **Dial:** ⭐ **addressing alone** — a read that stops at the addressed launch point, spending **zero** settling steps.
- **Laundering control:** `store_removed` on every row, **plus** the matched-bytes lookup floor (charter §C).
- **Falsifies:** that addressing alone carries the generation table.
- **Does NOT falsify:** losing to `knn_proto`, an unbounded real-valued channel (metric-native-ceiling theorem).

## ⛔⛔ WHAT THIS IS, AND THE NAME IT MUST NEVER BE GIVEN
The arm is `x0` — **request → address-book posterior argmax over the written atoms → decode**. ⭐ **It is an
ADDRESSED READ.** ⛔⛔ **It is NEVER described as dynamics, as settling, as a landscape read, or as "the dynamical
read at zero steps".** ⛔ It spends no steps, it uses no coupling, and it has no energy trace. **Calling it dynamics
would be the single most damaging overclaim available to this programme right now.**

## ⛔⛔ THE CORRECTION THIS TASK EXISTS TO NAIL DOWN — read before planning
The Hub has been relaying *"the launch beats the codebook read 0.6852 vs 0.5949 on 3/3 seeds."*
⭐ **That is TRUE ONLY AT 8× THE BYTES.** Measured (`gen-read-cell-arm`, diagnostic, 3 seeds):
| rung | store bytes | launch `x0` | vs `clu` 0.5949 ± 0.0082 |
|---|---|---|---|
| **A** | **524,288 — MATCHED to `clu`** | **0.5008 ± 0.0498** | ⛔ **LOSES by 0.094** |
| B | 4,194,304 — **8×** `clu` | 0.6852 ± 0.0173 | wins, ⛔ **not byte-matched** |
⇒ ⛔⛔ **THE HEADLINE IS THE MATCHED-BYTES ROW, AND TODAY IT IS A LOSS.** ⛔ The 8× row may be reported only as a
declared-larger companion, always with its byte multiple attached. ⛔ **Quoting 0.6852 against 0.5949 without the
byte ratio is forbidden.**

## The one job
**Pre-register the addressed read as an arm, then score it on composition-B at MATCHED DECLARED BYTES.**

## Scope
1. ⭐ **`PREREG.md` FIRST**, before any scored number: the arm, its byte ledger, its like-for-like rows, the bars,
   3 seeds, and **what result would prove it wrong**. ⛔ Filed after = void.
2. ⭐⭐ **ESTABLISH THE BYTE LEDGER — this is the crux, ⛔ not a formality.** The read uses the atom store and the
   address book. **Does it need the low-rank coupling at all?** If it does not, its ledger is **store only** —
   half the dynamical read's, and **exactly `clu`'s 524,288 B at rung A.** ⭐ **Assert the ledger in code and abort
   on mismatch.** ⛔ Do not assume; measure `n_bytes()` of the live objects.
3. **Score on composition-B aggregate, external classifier, the shipped cell**, same seeds, same split — ⛔ the
   frozen harness at the pin, read-only. Report `distinct_factor_tuples_B` beside the aggregate.
4. **Like-for-like rows, all of them:** ⭐ **atom-mean lookup at matched bytes** (the Advisor's named rival) ·
   `store_removed` · `clu` (the codebook read) · `ebm_compose` · `lace` **0.3117 ± 0.0153** *(cell harness; ⛔ not
   the 0.3241 baselines row)* · `knn_proto` **0.8650 ± 0.0122** as ceiling.
5. ⛔ **The charter §C floor applies:** the arm must beat a **matched-bytes lookup table**, or the store is
   decorative and the result does not count. **Report it either way.**

## ⛔ What is NOT in scope
⛔ No ladder, ⛔ no tuning rungs, ⛔ no settling, ⛔ no temperature, ⛔ no second arm. ⭐ **One arm, two byte points,
one table.** If it loses, ⭐ **it loses and is reported as the FLOOR for target 1** — that is a useful result and the
reason this spoke exists.

## Files you own — and nothing else
- `chlu/experiments/exp_gen_cell.py` (**the new arm only**) · `tests/test_gen_cell.py` · `.claude/outputs/gen-addressed-read/**`
⛔ Not `chlu/core/gen_landscape.py` (frozen — you *call* it), ⛔ not `chlu/config.py`, ⛔ not the classifier,
splits, rival rows, byte-ledger or bypass assertions — **those are the pin and editing them voids your numbers.**

## Acceptance
1. `PREREG.md` filed before any scored number.
2. ⭐ **The byte ledger measured and asserted**, with a plain statement of whether the coupling is needed.
3. ⭐⭐ **The matched-bytes verdict, led with** — WIN / TIE (favourable) / TIE / LOSS / INDETERMINATE by §2b as
   hardened (`WIN` needs `Δ > m`, `m = 0.020`; a tie counts only against the ceiling).
4. The 8× companion row, **with its byte multiple on the same line**.
5. The matched-bytes lookup floor, reported.
6. Targeted tests green, **passed / skipped / failed**. ⛔ No full suite. A **cost line** (⭐ CPU; ⛔ no GPU —
   this is a floor measurement, not a headline run).

## What would prove this spoke wrong
- ⭐⭐ **You led with 0.6852.** That is the 8× row. Leading with it is the overclaim this task was written to stop.
- ⭐ **You called it dynamics**, or let a reader think settling contributed. It spends zero steps.
- ⭐ **You assumed the byte ledger** instead of measuring it, so "matched bytes" is an assertion rather than a fact.
- ⭐ **It fails to beat a matched-bytes lookup table and you reported it as a win anyway.**
