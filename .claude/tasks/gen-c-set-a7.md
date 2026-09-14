# Task: gen-c-set-a7 — close the term-count leak, and make the class of bug impossible to repeat

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-c-set-a7.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (all) · **`.claude/tasks/gen-clu-phase1-rulings.md` — R26 ruling 1 is the brief** · `.claude/outputs/gen-certificate.md` **§3 and §5** (where the leak was found and what A7 is) · `chlu/data/gen_splits.py::build_c_invalid`.
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Base:** `main @ 3faef27` · **Branch:** `agent/experiment-engineer/gen-c-set-a7` · **Worktree:** fresh (`../CHLU-gen-a7`).
- **Env:** main venv, ⛔ no `uv sync` in the worktree. ⛔ CSF3 unreachable (2FA) — local CPU only.

## ⭐ DIAL DECLARATION
- **Dial:** *none — instrument repair.* ⛔ No store, no arm, no CLU number.
- **Laundering control:** the no-model detector **is** the control, and it is the deliverable.
- **Falsifies:** the detector still separates C from A∪B above chance after the fix ⇒ another structural giveaway remains.

## The one job
Ship **A7** and make the **no-model detector a suite test** that every future query-set definition inherits.

## The leak, so you do not re-introduce it
All **36** `duplicate_factor` C records carry **7 terms**; A and B carry **6**. A function that counts terms scores
**AUROC 1.0000** on C-dup-vs-B and **0.7500 pooled** — with no model in it. ⛔ Every arm inherits it, and it is
invisible in an aggregate C number. **Cause: the C spec never required C records to be structurally
indistinguishable from A/B.**

## 1. A7 — the fix (from `gen-certificate` §5)
Each **valid** request gets one **redundant** extra term: a factor repeated **at the value it already has**. It is
still satisfiable (it is the same image) and it is **7 terms long**. ⭐ **Built from valid requests alone**, so no
invalid example is used and the zero-invalid-supervision claim is untouched.
Emit A7 alongside the existing sets, checksummed, same single-source discipline. ⛔ Do not modify the C records.

## 2. ⭐ The no-model detector — a SUITE TEST, not a convention (R26, both riders)
> *"The convention that has to be remembered is the one that failed."*
- ⭐ **It runs on the REGISTERED query-set artifact**, discovered from disk — ⛔ **never a hardcoded list.** Every future
  C definition inherits the test automatically. That is the whole point; a test that only covers today's C set is a
  restatement of the bug.
- It may use **every field EXCEPT factor values**: term count, term ordering, id ranges, record length, key order,
  any structural property you can reach. ⭐ **Be adversarial — you are trying to break our own benchmark.**
- **Assert `AUROC ≤ 0.5 + tolerance` against A∪B**, tolerance **declared in the test**, and **fail loudly**.
- Report the realised detector AUROC **before and after** A7, per kind (`duplicate_factor`, `out_of_range`) and pooled.

## 3. `data_root` must fail loudly (R26 ruling 7)
`gen_data.data_root` means the *dataset* dir for `shapes3d_root()` and its *parent* for `default_gen_data_root()`.
`exp_gen_store.load_c_records` uses the parent spelling, so a wrong `--data-root` logs *"no C set … SKIPPED"* and
continues. ⛔ **A silently skipped C set is "declared not-run reported as null" in code form.** Make it **raise**, with
**no default that can paper over it**, and cover it with a test that asserts the raise.

## Files you own
`chlu/data/gen_splits.py` (A7 builder, additive) · `chlu/cli/data_cmd.py` (emit A7) · `chlu/experiments/exp_gen_store.py`
(**the `load_c_records` raise only** — ⛔ touch nothing else in that file) · `tests/test_gen_c_set_a7.py` (new) ·
`chlu/config.py` (fields inside the **existing `GenDataConfig`** block only).
⛔ Nothing under `chlu/core/`, `chlu/eval/`, `scripts/csf3/`.

## Acceptance
1. A7 emitted for Shapes3D (and dSprites if free), checksummed, `|A7| = |A|` convention stated.
2. ⭐ The detector test runs on the **registered artifact**, uses non-value fields adversarially, and **fails loudly**
   above tolerance. Realised AUROC reported before/after A7, per kind and pooled.
3. `load_c_records` raises on a wrong `data_root`; test asserts it.
4. ⛔ C records unmodified; their checksums reproduce bit-identically — asserted.
5. Suite green (before/after, main venv); ref verified from the main repo before worktree removal.

## What would prove this spoke wrong
- ⭐ **Your detector passes because you only tried term count.** Try ordering, id ranges, record length, key order.
  The next giveaway will not be the one we already found.
- The test hardcodes today's C set, so the next C definition silently escapes it.
