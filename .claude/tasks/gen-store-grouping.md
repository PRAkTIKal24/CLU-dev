# Task: gen-store-grouping — group channels into wells, learn modes, and prove the read cannot bypass the store

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-store-grouping.md` + artefacts under `.claude/outputs/gen-store-grouping/`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (all) · **`.claude/tasks/gen-clu-phase1-rulings.md`** (binding: R1, R2, R5, R6) · `.claude/outputs/gen-feature-bridge.md` (your input) · `chlu/core/factored_store.py` and `chlu/core/multiwell_read.py` (the existing well/mode machinery — **reuse it, do not re-invent it**) · `chlu/data/gen_splits.py`.
- ⛔ Read no campaign ledger, charter or handover — **EXCEPT `.claude/handover_context.md` §7 Known Issues, which is required reading** (§7 only; not the running log).
- **Base:** `main`, after `gen-feature-bridge` merges · **Branch:** `agent/experiment-engineer/gen-store-grouping` · **Worktree:** fresh (`../CHLU-gen-store`). ⛔ `CHLU-bprime-family2`, `CHLU-wt2`, `CHLU-wt3`, `CHLU-merge` are parked, not free.
- ⛔ **PRECONDITION — mechanical.** `git cat-file -e main:chlu/core/gen_bridge.py` succeeds **and** the bridge's
  feature-cache manifest exists **and** its gate verdict reads `PROCEED`. If the verdict reads `STOP`, **do not start** —
  the phase is halted and that is a Head decision.
- **Env:** `uv sync --extra gen`. **CLI contract (binding):** your module accepts `--dataset --pretext --stage --arm --seed --out`.

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** *isolation* — whether a factored store can be addressed and read without the read path routing around it.
- **Laundering control:** ⭐ **random grouping at the same K**, run as a full arm on every number, never as a footnote.
- **Falsifies:** random grouping matches correlation clustering within seed spread ⇒ the grouping carries no structure
  and the wells are arbitrary. **Also falsifying: deleting the store leaves the read output unchanged** (see below).
- **Does NOT falsify:** a low absolute read accuracy that is low for both groupings — that is a hard dataset or a weak
  feature cache, and the *contrast* is what is under test.

## The one job
Group the cached feature channels into `K` wells, learn the modes per well, and train the addressing/read path with a
**store-only objective** — such that the correlation-clustered grouping beats random grouping at the same `K`.

## ⛔ The failure this whole programme exists to catch
This programme's standing result is that **the block routes around the memory** — deleting the entire store changed the
loss by under 0.001 bits. ⭐ **So the store-only discipline is not hygiene here, it is the point: in this phase NO
encoder-to-decoder bypass exists.** Build it so a bypass is impossible, then *measure* that it is impossible:
- **Assert structurally** that no path carries encoder features to the decoder except through the store read.
- **Measure it:** delete the store and re-run the read. ⭐ **If the read output does not change, you have a bypass and
  the task has failed** — report that, loudly, as the finding. Do not repair it and re-run silently.
Report the delete-the-store delta as a number, not as a claim.

## What to build
1. **Correlation clustering** of channels into `K` wells on the **cached train activations only** (fixed arm).
   Declare `K` and how it was chosen; ⛔ `K` must not be tuned against anything held out.
2. **Random grouping control at the same `K`**, same seeds, same everything else.
3. **Modes learned per well**, reusing `factored_store` / `multiwell_read` where they fit.
4. **Store-only read objective.** No bypass, per above.
5. **Store byte count**, reported exactly — `gen-baselines` byte-matches the kNN arm to it and cannot start without it.
   Emit it as a machine-readable field, not prose.
6. **The well-space C mapping (R2 addendum).** After grouping, report how the **factor-space** C queries land on wells.
   ⛔ **This is a reported diagnostic for grouping quality. It NEVER becomes the query set** — C stays factor-space and
   identical for every arm, because two of the four arms have no wells at all.

## ⛔ Standing convention
**Wells are never named semantically** — not in code, tests, artifacts or figure captions. Integer indices only.

## Files you own
`chlu/core/gen_store.py` · `chlu/experiments/exp_gen_store.py` · `tests/test_gen_store.py` — all new.
`chlu/config.py` — append one dataclass + the **three-site** wiring (dataclass, `load_config`, `save_config`; the house
convention, as `gen-data-splits` established). `chlu/cli/experiment_cmd.py` — append one subparser. ⛔ Touch no existing line.
⛔ You own nothing under `scripts/csf3/`, `chlu/eval/`, or `chlu/core/gen_bridge.py`.

## Acceptance
1. Correlation grouping vs random grouping at the same `K`, **≥3 seeds, with spread**, on the read metric you declare
   up front. State the margin you predicted before running (protocol §5 pre-registration rule) and whether it held.
2. **The delete-the-store delta, reported as a number.** A near-zero delta is a failed task reported as such.
3. Store byte count emitted machine-readably.
4. Well-space C mapping reported as a diagnostic.
5. Suite green: pass count before and after, **main venv** per protocol §4, `uv sync --extra gen`.
6. `git log --oneline main..agent/experiment-engineer/gen-store-grouping` re-verified **from the main repo** before
   any worktree removal.

## What would prove this spoke wrong
- **Deleting the store changes nothing.** That is the programme's known failure mode arriving one stage earlier, and it
  is far more valuable caught here than at cell time. Measure it before you believe any read number.
- Random grouping matches the clustered arm — then `K` wells are `K` arbitrary partitions and the addressing claim is empty.

---

# ⟲ AMENDMENT 2026-09-10 — base moved, gate 1 passed, and two things folded in before spawn

## Base and precondition
**Base is now `main @ fcdca43`.** ⭐ **Gate 1 verdict = `PROCEED`** (`ceiling_A` 0.9507, `ceiling_B` 0.8472,
ratio 0.891 vs floor 0.8, 3/3 seeds). Your precondition is **MET**. `chlu/core/gen_bridge.py`,
`chlu/eval/gen_probe.py` and `chlu/experiments/exp_gen_bridge.py` are on `main`. ⛔ Do not rebuild any of them.

## ⛔ SUBMIT-LINE CONSTRAINTS — the Hub writes these, but know why they are there
Every gen submit line carries `PAYLOAD_MODULE=chlu.experiments.exp_gen_<stage>` (without it the job aborts loudly)
and `ARMS=clu` for single-arm stages (without it the launcher silently spends 3/4 of the budget and creates four
output directories). Also `GEN_GIT_COMMIT=$(git rev-parse HEAD)`.

## ⚠ The instrument is saturated — do not read it as a difficulty axis
The probe reads **real** images at **1.0000** on both A and B. ⭐ **Every accuracy loss in this phase is the decoder's,
not the probe's.** ⛔ Do not attribute any gap to "a hard probe".

## ⭐ What the bridge found that changes what you should look at
Montero's held-out cell requires **all three hues ≥ 0.8**, which collapses object/background contrast. Measured against
a photometrically matched **seen** control: the **`scale`** deficit on B (0.708) is indistinguishable from the same
deficit on seen data (0.703) — it is a *rendering* artefact. ⛔ **The only genuinely compositional factor is `shape`,
whose ceiling on B is 0.500.**
**What this means for you:** when you report the read metric and the delete-the-store delta, ⭐ **report `shape`
separately from the aggregate.** The aggregate is dominated by four factors sitting at ≈1.0 that carry no
compositional information, so an aggregate-only number can hide the thing under test. This costs you one extra column.
⛔ You still do **not** run B or C and still render **no** kill-rule verdict — that is `gen-cell-run`'s job.

## Environment reminders that cost the last spoke time
- ⛔ **No `uv sync` inside your worktree** (w6 lesson) — reuse the main venv:
  `PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python -m …`, with `CHLU_GEN_DATA_ROOT` set (worktrees
  have no `.claude/`). h5py is in the main venv via the `gen` extra.
- ⚠ **§7.23, and it WILL bite you.** `test_friction_field` / `test_goldstone` / `test_lattice` / `test_twins` /
  `test_kt` / `test_wormhole` call `jax.config.update("jax_enable_x64", True)` **at module import**, and they sort
  before yours. Any `eqx.nn.Conv2d` fed an explicitly-float32 array then dies **in the full suite while passing
  alone.** ⭐ **The durable remedy is for your module to cast its input to its own parameter dtype** —
  `chlu/core/gen_bridge.py` is the shipped exemplar. ⛔ A float32-pinning pytest fixture is NOT the fix; it leaves the
  non-pytest path broken.
- ⚠ **CSF3 is unreachable from agent sessions** (no VPN/DNS). Run locally on CPU, and report any cluster item as
  **NOT RUN** with the submit line you would have used. ⛔ Never report an unrun job as a pass.

---

# ⟲ AMENDMENT 2026-09-10 — you are RUNG R0 of the factorisation ladder. Emit its x-axis.
A new experiment (the **factorisation ladder**, paper 1's mechanism section) plots three capabilities against
**leakage** across six grouping rungs. ⭐ **You are R0** — *fixed correlation-clustered grouping, plain reconstruction
encoder, label-free.* ⛔ **If you do not emit the x-axis, R0 has no x-coordinate and this whole spoke gets re-run.**
It is cheap now and expensive later, so it is folded in before you start.

## What to add — the leakage measurement, per well
**leakage = 1 − per-well probe transfer.** For each well: train a per-factor probe on **that well's sub-vector only**,
on **seen** combinations; test on **held-out**; ⭐ **normalise against the matched-seen control** (R16 — same hue
corner, seen shape), not against random-A. ⛔ The matched-seen row is a permanent companion, not an optional extra.
Report per well and pooled, **≥3 seeds with spread**, for **both** the clustered grouping **and** the random-grouping
control at the same K.

**Companion factorisation measures, same protocol:** **DCI**, **MIG**, and **post-clustering block-diagonality** of the
channel correlation matrix. These are descriptive companions to leakage; ⛔ none of them replaces it as the x-axis.

## ⛔ Pin K and the store config as a REUSABLE artifact
Every later rung must use an **identical** store configuration and **identical K and byte count** — the ladder's whole
logic is that only the grouping changes. ⭐ So emit the store config as a machine-readable, loadable artifact that
later rungs consume verbatim, not as a number in prose. ⛔ A later rung silently re-deriving K voids the comparison.

## What does NOT change
Your one job, your acceptance criteria and your falsifiers are unchanged. The delete-the-store delta is still the
thing this spoke lives or dies on. ⛔ Still no B, no C, no kill-rule verdict.
