# Task: gen-arms-a7-wiring — make the rivals scorable under A7, and stop three wrong defaults

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-arms-a7-wiring.md` + artefacts under `.claude/outputs/gen-arms-a7-wiring/`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (all) · **`.claude/tasks/gen-clu-phase1-rulings.md` — R26 ruling 3, R27 (all four), R28** · **`.claude/outputs/gen-baselines.md` §0, §3, §4, §6c — it names every defect you are fixing, with line-level evidence** · `.claude/outputs/gen-c-set-a7.md` (the A7 registry and how to read it).
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Base:** `main @ 5fee3f4` · **Branch:** `agent/experiment-engineer/gen-arms-a7-wiring` · **Worktree:** fresh (`../CHLU-gen-a7wire`). ⛔ `CHLU-bprime-family2`, `CHLU-wt2`, `CHLU-wt3`, `CHLU-merge`, `CHLU-gen-*` are parked or live.
- **Env:** main venv, ⛔ no `uv sync` in the worktree. ⛔ CSF3 needs 2FA and is unreachable — local CPU only, emit the submit line.
- ⚠ **Budget: the refusal re-run is ~110 min for 3 seeds.** ⭐ **Write results per seed as they complete and skip any
  seed whose output exists** — a detached spoke can keep running and still fail its last write. ⛔ Never hold to the end.

## ⭐ DIAL DECLARATION
- **Dial:** *none — competitor arms brought up to the rulings.* ⛔ No CLU number, no kill-rule verdict.
- **Laundering control:** N/A — **you are maintaining the control.**
- **Falsifies:** N/A for a dial. Your falsifier: a rival that gets *weaker* because of a change you made.

## The one job
Make the kNN and LACE arms **scorable under A7**, and fix the three defaults that currently make any future run wrong.

## Why this exists — it is the Hub's scoping miss, not the analyst's
`gen-baselines-consolidate` was scoped as an analyst re-score. It could not be done: `exp_gen_baselines`,
`gen_knn.py`, `gen_lace.py` and `gen_request.py` contain **zero** references to `a7`/`redundant`, and **no
per-request refusal score is persisted** — so there is nothing on disk to re-score. ⭐ The analyst reported it as
**NOT RUN with proof** rather than inferring a number, which is the correct outcome. The fix is code.

## The four items
### 1. ⭐ A7 wiring — the refusal comparison depends on it
Score both arms' refusal against the **A7** valid pool as well as the ordinary one. ⭐ **Persist a per-request
refusal score** so any future re-score is a spreadsheet, not a re-run — that absence is what blocked this one.
⚠ **Half the work is already correct and must not be disturbed:** A7's registry leaves `out_of_range`'s control at
the ordinary 6-term valid pool, so **`out_of_range` is A7-clean as shipped** (kNN **0.8045 ± 0.1227**, LACE best
**0.4694 ± 0.1368**). ⛔ Only the `duplicate_factor` half and the pooled number need the new scoring. **Reproduce the
`out_of_range` numbers bit-identically as your regression check.**

### 2. ⛔ `include_coupling_bytes: bool = True` is now a WRONG default
`chlu/config.py::GenBaselinesConfig`. R26 ruling 3 removed the coupling field from the deployed store, so the budget
is **524,288 B**, not 688,128 B. Any future run silently inherits the wrong budget. **Flip the default to `False`**,
keep the flag for reproducing the historical run, and **assert the realised budget in the manifest.**

### 3. ⭐ R27.3's whitening gate fires on 2 of 3 seeds — run the fallback
The pre-registered selection picked a **non-exact** whitening on seed 0 (`1e-3`, |diag−1| = 0.2382) and seed 2
(`ledoit_wolf`, 0.1438); `prior_identity_near_exact = false` on both. ⛔ **A rival weakened by our own selection
procedure is our fault.** Nearest-exact fallback configs already exist in the trial grid with valid-only scores
(`gen-baselines.md` §6c) — **decode them**, or run the general form with the prior score included.
⭐ **Report both and quote the stronger.** ⛔ Never-weaken-a-competitor applies to our selection code.

### 4. C reporting (R27.4)
⛔ **By kind always** (`duplicate_factor`, `out_of_range`); pooled is a companion only. Report the contaminated and
A7-controlled columns side by side, ⛔ never merged into one.

## ⛔ What you must NOT do
- ⛔ Do not touch the composition cell. ⭐ It is **byte-budget-invariant** — `prototype_direct_read` never touches the
  byte-sized store and the prototype table (475,136 B) is inside **both** budgets, so the Head's mandatory-re-run
  trigger is **provably a no-op there**. The 31 % advantage lives entirely in the refusal cells.
- ⛔ Do not re-run composition, the bridge, the store or the decoder. Refusal only.
- ⛔ Do not weaken either rival to make a number move. If a change makes a rival worse, **stop and report**.
- ⛔ No kill-rule verdict — `gen-cell-run`'s job.

## Files you own
`chlu/eval/rivals/gen_knn.py` · `chlu/eval/rivals/gen_lace.py` · `chlu/core/gen_request.py` ·
`chlu/experiments/exp_gen_baselines.py` · `tests/test_gen_baselines.py` ·
`chlu/config.py` — **the existing `GenBaselinesConfig` block only.**
⛔ Nothing under `chlu/core/gen_store.py`, `gen_bridge.py`, `gen_compose.py`, `gen_certificate.py`, `chlu/data/`, `scripts/csf3/`.

## Acceptance
1. Both arms scored against A7; **per-request refusal scores persisted at the PINNED path**
   `.claude/outputs/gen-arms-a7-wiring/refusal_scores.json`, with a top-level `"status": "complete"` key written
   **last**, after every seed. ⭐ Downstream spokes gate on that file and that key — a different path breaks them.
2. `out_of_range` reproduces **bit-identically**; `duplicate_factor` and pooled reported in both columns.
3. `include_coupling_bytes` default `False`; realised budget **524,288 B** asserted in the manifest.
4. LACE whitening fallback run on seeds 0 and 2; both configurations reported, **the stronger quoted**.
5. Per-seed incremental writes; any seed not completed reported as **NOT RUN**.
6. Targeted tests green (`tests/test_gen_baselines.py` + the A7 tests). ⚠ Full suite is run **once at wave close** by
   the Hub — report your targeted result and your base commit, ⛔ do not start a full suite.

## What would prove this spoke wrong
- ⭐ **A rival got weaker.** You are maintaining the control; if your change lowers a rival's number, you have
  hobbled it. Check `out_of_range`'s bit-identity first — it is the regression that catches this.
- You re-scored by inferring from aggregates instead of persisting per-request scores. The next re-score would be
  blocked exactly as this one was.
