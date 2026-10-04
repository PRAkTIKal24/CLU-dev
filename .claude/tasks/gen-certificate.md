# Task: gen-certificate — a refusal criterion derived from the store, with provably zero invalid-example supervision

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-certificate.md` + artefacts under `.claude/outputs/gen-certificate/`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (all) · **`.claude/tasks/gen-clu-phase1-rulings.md`** (binding: **R2 and R7**, plus R5/R6) · `.claude/outputs/gen-store-grouping.md` (your input) · `chlu/core/soft_certificate.py` (existing barrier/reach machinery — **reuse what fits, do not edit it**).
- ⛔ Read no campaign ledger, charter or handover — **EXCEPT `.claude/handover_context.md` §7 Known Issues** (§7 only).
- **Base:** `main`, after `gen-store-grouping` merges · **Branch:** `agent/experiment-engineer/gen-certificate` · **Worktree:** fresh (`../CHLU-gen-cert`).
- ⛔ **PRECONDITION — mechanical.** `git cat-file -e main:chlu/core/gen_store.py` succeeds and the store artefact exists.
- **Env:** `uv sync --extra gen`. **CLI contract (binding):** `--dataset --pretext --stage --arm --seed --out`.
- ⚠ **Runs in parallel with `gen-dream-sampler` and `gen-baselines`** — all three branch off the same base and share no files.

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** *admission* — whether a request can be refused from the store's own geometry.
- **Laundering control:** ⭐ **a plain distance-to-store threshold** on the same features. If naive geometry does as
  well, the derivation adds nothing. (The byte-matched kNN arm is `gen-baselines`'; you build the trivial in-house
  version as your own control so you find this out before the cell does.)
- **Falsifies:** certificate AUROC at chance; **or any threshold that touched invalid data.**
- **Does NOT falsify:** losing to a **label-supervised** post-hoc detector. LACE's refusal energy comes from a
  factor-labelled classifier; ours uses zero labels. That comparison is reported from phase 1 onward but is **not** a
  kill rule (R7).

## The one job
Compute, for a requested factor combination, a composition energy/barrier **from quantities the store already exposes**
(energies, Hessians, reach) — and report how well it separates the impossible requests from the possible ones.

## ⛔ The supervision rule — this is the claim, so it is asserted, not promised
- **No learned refusal head.** The criterion is derived, not fitted to a refusal target.
- **ZERO invalid-example supervision anywhere.** No synthesised negatives. No threshold tuned on invalid queries.
- **Thresholds may be set on VALID data only** — e.g. a quantile of seen-combination barriers.
⭐ **Make this mechanically checkable rather than a promise in prose:** route every threshold-fitting call through one
function that records a hash of exactly the data it saw, and **assert that no C record's id ever appears in it.**
Report the assertion and the count. A reviewer will not take our word for this, and neither should we.

## The metric — read R7 before you write any AUROC
B is **72 combinations of 480,000** — **0.0150%** of A∪B. An arm can therefore refuse all of B at almost no cost on a
pooled A∪B-vs-C AUROC, which is exactly the penalty that comparison was meant to impose. **So report all three,
separately and with equal prominence:**
1. **A-vs-C** · 2. ⭐ **B-vs-C** · 3. class-balanced **A∪B-vs-C**
plus the risk-coverage curve and **AURC**. ⛔ Do not report a single pooled AUROC as "the" number.
⚠ Whether kill rule (iv) binds on B-vs-C is **an open Head ruling**. Report all three; ⛔ **render no kill-rule verdict** —
that is `gen-cell-run`'s job, against a filed prereg.

## Files you own
`chlu/core/gen_certificate.py` · `chlu/eval/gen_cert_eval.py` · `tests/test_gen_certificate.py` — new.
`chlu/config.py` (append + three-site wiring) · `chlu/cli/experiment_cmd.py` (append one subparser). ⛔ No existing line.
⛔ **Do not edit `chlu/core/soft_certificate.py`** — it is shared with a live campaign. Import from it; if it needs a
change, stop and report what and why.

## Acceptance
1. Certificate computed for the A, B and C request sets, **≥3 seeds, with spread**. C is read from
   `.claude/data/gen/shapes3d/c_invalid.json` — ⛔ single-sourced, byte-identical to every other arm, never regenerated.
2. All three AUROCs + risk-coverage + AURC reported separately.
3. ⭐ **The no-invalid-supervision assertion runs and its count is reported.** This is the criterion the task lives on.
4. The distance-to-store laundering control reported alongside every certificate number.
5. Suite green (before/after, main venv); ref verified from the main repo before worktree removal.

## What would prove this spoke wrong
- **A threshold saw a C record.** Then the headline claim is void regardless of how good the AUROC looks. The assertion
  is what protects the result from us.
- The plain distance threshold matches the derived barrier — then the derivation is decoration over naive geometry.

---

# ⛔ STANDING CONSTRAINT 2026-09-10 — CSF3 CANNOT BE REACHED FROM AN AGENT SESSION
**Its ssh requires 2FA. No spoke has ever submitted a job and none ever will.** ⛔ Do not try, and ⛔ **never report a
cluster run you did not see complete.**
**What you do instead:** run locally on CPU at whatever scale fits, and **emit the exact submit line** the Head will
run — `PAYLOAD_MODULE=chlu.experiments.exp_gen_<stage>`, `ARMS=…`, `GEN_GIT_COMMIT=$(git rev-parse HEAD)`, and every
defaultless axis (`DATASET`, `PRETEXT`, `STAGE`) — plus the realised output paths it will produce. Report every
cluster acceptance item as **NOT RUN** with that line beside it. ⭐ A validated dry run plus a correct submit line is
the *complete* deliverable here; it is not a partial one.

## Environment reminders that have already cost two spokes time
- ⛔ **No `uv sync` inside your worktree** (w6 lesson) — reuse the main venv:
  `PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python -m …`, with `CHLU_GEN_DATA_ROOT` set (worktrees have
  no `.claude/`). h5py is in the main venv via the `gen` extra.
- ⚠ **§7.23 will bite any conv module you write.** `test_friction_field` / `test_goldstone` / `test_lattice` /
  `test_twins` / `test_kt` / `test_wormhole` set `jax_enable_x64=True` **at module import** and sort before yours; an
  `eqx.nn.Conv2d` fed an explicitly-float32 array then dies **in the full suite while passing alone**.
  ⭐ **Fix it by casting the input to the module's own parameter dtype** — `chlu/core/gen_bridge.py` is the shipped
  exemplar. ⛔ A float32-pinning pytest fixture is not the fix; it leaves the non-pytest path broken.
- ⚠ **The probe is saturated** — it reads *real* images at **1.0000** on both A and B. Every accuracy loss in this
  phase is the decoder's. ⛔ Never attribute a gap to "a hard probe".

---

# ⟲ AMENDMENT 2026-09-11 — read this before you write a line. Base is now `main @ 9a4457d`.

## ⛔⛔ THE TRAP THAT WOULD VOID YOUR ENTIRE RUN
`chlu/config.py` line ~3358: **`dec_recipe: str = "baseline"`.** That default is the **SUPERSEDED** decoder — the one
that **FAILED** the gate (balanced accuracy **0.7191**, 0/3 seeds clear). The winner is **`v1_capacity_schedule`**
(**0.9414**, 3/3 clear), and the default was deliberately left alone so Montero's historical behaviour reproduces.
⭐ **You must select the winning recipe explicitly, and you must ASSERT the decoder's parameter hash against
`gen-decoder-lift`'s winner before recording any number.** ⛔ An arm decoded through the baseline recipe is void, and
it will look perfectly normal in the logs. **Check this first, not last.**

## Configuration: build against Montero, parameterise for wide
The cell will ultimately run on the **widened** configuration (Advisor ruling R23.2). ⛔ **Do not hard-code Montero
paths, cell counts or split names.** Build and validate against the **Montero** configuration — it exists, it is fully
validated, and it is what the store was fitted on — but take the condition as a **parameter** (`--condition`, the
passthrough `gen-bridge-wide` already shipped) so the same code re-points at `recomb2element_wide` without an edit.
⭐ **A spoke that hard-codes the configuration costs a full rebuild later.**

## Metrics, as ruled (R23) — ⛔ do not invent your own
- **Composition: BALANCED ACCURACY is the gate statistic. AUROC is a mandatory companion and is NEVER the gate.**
- **Refusal: the primary is AUROC(B unseen-valid vs C impossible).** C-vs-A and C-vs-(A∪B) reported alongside.
- ⭐ **Two ceilings on every table you produce:** the **decoder ceiling** (reachable from real features) **and the
  STORE ceiling** — the in-corner store read is **0.6866** against a **0.9414** real-feature ceiling. ⛔ A number
  quoted against only one of the two is incomplete. The 30-bit channel is **not** touched in phase 1.
- ⛔ Shared-corner cells (`montero`, `wide-000`) are a **two-class corner, never pooled**, if you touch wide cells.
- ⚠ The C set is now sized `|C| = |B|` (Shapes3D **72**). ⛔ Read `c_invalid.json` from disk; never regenerate it.
- ⚠ **`shape` is the compositional term.** Report it separately from the aggregate, always. Four of six factors sit at
  ≈1.0 and carry no compositional information.

## ⛔ You are running UNATTENDED, overnight
- **Write results incrementally.** ⛔ Do not hold everything to the end — a crash at hour 6 must not lose hours 1–5.
- **Log progress with timestamps** so elapsed time is attributable afterwards.
- ⭐ **If you run out of time, a truthful partial is the deliverable.** Report exactly which seeds/arms completed and
  which did **NOT RUN**. ⛔ Never a zero, never a silent 1-seed number presented as 3.
- ⛔ **No `uv sync` in your worktree** (w6). Main venv: `PYTHONPATH=<worktree>
  /Users/user/Desktop/CHLU/.venv/bin/python -m …`, `CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen`.
- ⚠ **§7.23:** six test modules set `jax_enable_x64=True` at module import and sort before yours; cast inputs to your
  module's own parameter dtype (`chlu/core/gen_bridge.py` is the exemplar). A float32-pinning fixture is NOT the fix.
- ⛔ **CSF3 needs 2FA and is unreachable.** Local CPU only; emit the submit line; report cluster items as **NOT RUN**.

## Worktrees — three of you are running at once
Fresh worktree each; ⛔ `CHLU-bprime-family2`, `CHLU-wt2`, `CHLU-wt3`, `CHLU-merge`, `CHLU-gen-store` are parked.
**`chlu/config.py` is shared by all three of you.** Append your own dataclass at the **end of the file**, below every
existing block; ⛔ touch no other hunk. Rebase onto `main` before finishing and resolve **only** your own.

## ⛔ `chlu/config.py` — FOUR spokes are appending to the same anchor. Read this before you rebase.
`gen-dream-sampler`, `gen-certificate`, `gen-baselines` and `gen-r0-wide-refit` are all running **now**, and all four
were told to append at the **end of `chlu/config.py`** (currently 3,930 lines, ending in `save_config`). ⭐ **You WILL
hit a conflict there when you rebase. It is expected, it is not a defect, and it is not a reason to stop.**

**How to resolve it — the only acceptable resolution:**
- ⭐ **Keep BOTH blocks**, in whatever order they arrive. Your block and the other spoke's block are independent
  dataclasses; there is no semantic conflict, only a textual one at the same anchor line.
- ⛔ **NEVER delete, rename, reorder or "tidy" another spoke's block to make the conflict go away.** That silently
  destroys a peer's work and it will not be caught until their tests fail on `main`.
- ⛔ Do not touch any hunk above the anchor. Everything from line 1 to `save_config` belongs to earlier waves.
- If the conflict is anything other than "two additions at the same end-of-file point", **stop and report** — that
  means someone edited shared state and it is not yours to adjudicate.

**Wiring:** if your dataclass needs to round-trip, wire it at the three house sites (the dataclass, `load_config`,
`save_config`) exactly as `GenDataConfig` and `GenBridgeConfig` do. ⚠ If you only add **fields to an existing
dataclass**, no new wiring is needed — they round-trip automatically.
