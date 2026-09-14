# Task: gen-dream-sampler — conditional dreaming by clamped-well Langevin, with an audit trail that can convict it

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-dream-sampler.md` + artefacts under `.claude/outputs/gen-dream-sampler/`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (all) · **`.claude/tasks/gen-clu-phase1-rulings.md`** (binding: R3, R5, R6, **R9.5 on Metropolis**) · `.claude/outputs/gen-store-grouping.md` (your input) · `chlu/experiments/exp_c_dreaming.py` and `chlu/core/integrators.py` (existing Langevin machinery — **reuse it**).
- ⛔ Read no campaign ledger, charter or handover — **EXCEPT `.claude/handover_context.md` §7 Known Issues** (§7 only). ⚠ §7.9 (FDT) and the `langevin_noise` flag are directly relevant to you.
- **Base:** `main`, after `gen-store-grouping` merges · **Branch:** `agent/experiment-engineer/gen-dream-sampler` · **Worktree:** fresh (`../CHLU-gen-dream`).
- ⛔ **PRECONDITION — mechanical.** `git cat-file -e main:chlu/core/gen_store.py` succeeds **and** the store artefact
  named in `gen-store-grouping.md` exists on disk **and** its delete-the-store delta is non-trivial. ⛔ If that delta was
  near zero, a bypass exists and there is nothing here worth sampling — stop and report.
- **Env:** `uv sync --extra gen`. **CLI contract (binding):** `--dataset --pretext --stage --arm --seed --out`.

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** *isolation* — whether clamping the wells a request addresses actually steers what comes out.
- **Laundering control:** ⭐ **the zero-step rollout** — decode the clamp directly, no Langevin. Run it on every number.
- **Falsifies:** the zero-step control matches the full rollout ⇒ **the sampler is inert and we are decoding the clamp,
  not sampling the landscape.** Also falsifying: energy traces flat, or the chain never leaves its initialisation.
- **Does NOT falsify:** samples that are blurry or low-fidelity in absolute terms — the frozen decoder's measured
  ceiling bounds that, and it is read as a ratio (R3.2), never absolutely.

## The one job
Implement conditional dreaming — clamp the wells addressed by the requested factors, relax the rest, run Langevin on
the landscape, decode through the **shared frozen decoder** — and emit an audit trail that lets a referee convict the
sampler if it is doing nothing.

## ⭐ The sampler ruling (binding, from the Head)
**Declare the sampler.** State whether a Metropolis correction is applied and, **if not, why it is unnecessary at this
scale.** ⛔ "We used Langevin" is not a declaration. Log **acceptance and energy traces** so the sampler can be audited
after the fact by someone who does not trust us.
⚠ Context you are given so you do not re-derive it: Du et al. (ICML 2023, arXiv:2302.11552) concluded that for composed
energies **the sampler, not the model, is the failure point.** Their critique bites hardest on uncorrected samplers over
*products of diffused* models. Whether it bites here is **yours to argue in writing, with the traces as evidence** —
not to assume either way.

## The zero-step control, and why it is the deliverable
A clamped-well rollout that simply decodes its own clamp will produce respectable accuracy on seen combinations while
performing no inference at all. ⭐ **So the acceptance criterion is the contrast, not the accuracy:** full rollout vs
zero-step, same clamp, same seeds. If they match, the sampler is inert and the whole generative story is decoration.

## What to build
1. **Clamped-well Langevin rollout** on the store's landscape: clamp the wells the request addresses, relax the rest.
2. **Decode through the ONE shared frozen decoder.** ⛔ Assert its parameter hash matches the bridge's. No arm may
   retrain or substitute a decoder — that would invalidate every cross-arm comparison in the phase.
3. **The store-removed ablation** — the identical path with the store disabled. You own this arm; the cell spoke runs it.
4. **Audit traces per rollout:** energy vs step, acceptance (or an explicit statement that none is computed and why),
   clamp drift, and the step count/step size actually used. Written to disk, not summarised in prose.
5. **A-cell smoke only** (seen combinations), reported as a fraction of the bridge's `ceiling_A`. ⛔ **Do not run B or C
   and do not report a cell number** — the cell belongs to `gen-cell-run`, after the prereg is filed.

## Files you own
`chlu/core/gen_dream.py` · `chlu/experiments/exp_gen_dream.py` · `tests/test_gen_dream.py` — new.
`chlu/config.py` (append + three-site wiring) · `chlu/cli/experiment_cmd.py` (append one subparser). ⛔ No existing line.
⛔ You own nothing under `chlu/eval/`, nor `gen_store.py`, `gen_bridge.py`, or anything in `scripts/csf3/`.

## Acceptance
1. **Full rollout vs zero-step control, ≥3 seeds, with spread.** Pre-register the separation you expect before running.
2. Sampler declaration written out: family, step size, steps, noise convention, Metropolis yes/no **with the argument**.
3. Audit traces on disk for every reported rollout, with paths listed.
4. Decoder-hash assertion passes; store-removed ablation runs.
5. A-cell accuracy as a fraction of `ceiling_A`. ⛔ No B, no C, no kill-rule verdict.
6. Suite green (before/after counts, main venv, `uv sync --extra gen`); ref verified from the main repo before worktree removal.

## What would prove this spoke wrong
- **Zero-step matches full rollout.** Then nothing is being sampled. Check this first, before you look at any accuracy.
- The energy trace is flat or monotone to a floor with no exploration — a chain that never mixes will still decode
  something plausible on *seen* combinations and tell you nothing about unseen ones.

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
