# Task: gen-baselines — the two competitor arms, built to win if they can

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-baselines.md` + artefacts under `.claude/outputs/gen-baselines/`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (all) · **`.claude/tasks/gen-clu-phase1-rulings.md`** (binding: **R7, R8, R9**, plus R5/R6) · **`.claude/outputs/gen-lace-spec.md` and `gen-lace-spec/lace_spec.json` — your build spec, every constant tagged with its provenance** · `.claude/outputs/gen-store-grouping.md` (for the byte count) · `chlu/eval/rivals/` (house conventions for competitor arms).
- ⛔ Read no campaign ledger, charter or handover — **EXCEPT `.claude/handover_context.md` §7 Known Issues** (§7 only).
- **Base:** `main`, after `gen-store-grouping` merges · **Branch:** `agent/experiment-engineer/gen-baselines` · **Worktree:** fresh (`../CHLU-gen-baselines`).
- ⛔ **PRECONDITION — mechanical.** `git cat-file -e main:chlu/core/gen_store.py` succeeds **and** the store byte count
  is present as a machine-readable field in `gen-store-grouping`'s artefacts. ⛔ Without the byte count the kNN arm
  cannot be matched, and an unmatched kNN arm is worthless — stop and report.
- **Env:** `uv sync --extra gen`. **CLI contract (binding):** `--dataset --pretext --stage --arm --seed --out`.
- ⚠ **Runs in parallel with `gen-dream-sampler` and `gen-certificate`** — same base, no shared files.

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** *none — competitor arms / instrument.* ⛔ No CLU number is produced here.
- **Laundering control:** N/A — **you ARE the control.**
- **Falsifies:** N/A for a dial. Your own falsifier is below, and it is the inverse of the usual one.

## The one job
Build the **kNN-over-features arm with distance-threshold refusal, byte-matched to the store**, and the **LACE-style
arm**, both running end-to-end on the shared frozen decoder.

## ⭐ ANTI-HOBBLING IS THE GOVERNING DISCIPLINE — and here it inverts
Normally a weak baseline costs us a fair fight. **Here it is worse than that.** The kNN arm is the comparator for
**two** kill rules — (ii) on B and (iv) on C. ⭐ **A weak kNN makes both pass trivially and lets a dead mechanism
through.** So a hobbled baseline does not cost us a fair fight; **it costs us the truth, in the direction that flatters
us.** Build both arms to win if they can. A rival that beats the CLU is the control working, and it gets reported
unsoftened.

## R8 — request embedding, declared once, applied identically
Impossible combinations have no images, so a request must be embeddable without one. ⛔ **Neither arm chooses its own.**
- **kNN:** composed **per-factor prototype features**, computed from the cached train features. You own these.
- **LACE:** composed **classifier energies**.
- The CLU arm's addressing head is `gen-store-grouping`'s. ⛔ Do not reimplement it or borrow from it.
- **Request sets A, B, C are single-sourced** from `chlu/data/gen_splits.py` and `c_invalid.json`, byte-identical
  across arms. ⛔ Never regenerate them.

## The kNN arm
- Distance-to-store refusal threshold, set on **valid data only** — same rule that binds our certificate.
- ⭐ **Byte-matched to the store: shrink to match, never grow, never re-solved by hand.** Report the realised byte
  ledger next to the store's. Total bytes as deployed; ⛔ no dtype normalisation.
- Sweep `k`, the metric and the normaliser as you would if you wanted it to win. Report the tuning-trial budget — it
  must equal every other arm's (Head's constraint).

## The LACE arm — build it per R9, which is already adjudicated
1. ⭐ **Whitening is mandatory.** Fit a shrinkage Gaussian (Ledoit-Wolf / ridge) over the cached train features; sample
   in `u = Σ^{-1/2}(x − μ)`. ⛔ **A bare `½‖x‖²` on raw post-ReLU features is a silent crippling and is refused** — the
   shipped ODE drift omits the prior score, and that omission is licensed *only* by an exact N(0,I) prior.
   ⛔ A learned flow prior is also refused: it breaks the invariance identity and stops being LACE.
   Shrinkage is mandatory where `D > N`. If `D` exceeds ~2e4, fall back to channel whitening **and declare it**.
2. **Both Langevin temperatures** (R9.4): as-published, and at `std = √(2·lr)`. The published constants are ~200× colder
   than their own target, so as-shipped LACE-LD is near-deterministic gradient ascent. **Report LACE's best.**
3. **Concessions granted to LACE, declared not hidden** (R9.3): dense factor labels, and a validation split for tuning
   the OR/NOT constants (`cof`, and the `0.1` in α). ⛔ Freezing them at the published 20/0.1 hobbles OR/NOT — the
   authors' own comments are literally *"these values work well"*.
4. ⛔ **Constant traps** (R9.6): shipped `n_epochs=50` (not the appendix's 100, not argparse's 200); batch 128/512, **not**
   the argparse default 64; the staircase decay schedule **never fires**; don't sweep `dis_temp`. OR is **non-commutative**
   and there is a hard **10-operand** limit.
5. Licence: NVIDIA **non-commercial**. Note it; vendor nothing without saying so.
6. ⛔ **`DO_NOT_USE_AS_A_BAR`** (R9.8): their FID 6.63 / ACC 0.972 / unseen-combination 0.94–0.98 are StyleGAN at up to
   1024² on their splits with their labels. **Only your re-implementation on our features and our splits is comparable.**

## Shared-decoder rule
Both arms decode through the **ONE frozen decoder** from `gen-feature-bridge`. ⛔ Assert its parameter hash. No arm
substitutes or retrains a decoder — that would void every cross-arm comparison in the phase.

## Files you own
`chlu/eval/rivals/gen_knn.py` · `chlu/eval/rivals/gen_lace.py` · `chlu/core/gen_request.py` (request embeddings) ·
`tests/test_gen_baselines.py` — new. `chlu/config.py` (append + three-site wiring) · `chlu/cli/experiment_cmd.py`
(append one subparser). ⛔ No existing line; ⛔ nothing under `chlu/core/gen_store.py`, `gen_dream.py`, `gen_certificate.py`.

## Acceptance
1. Both arms run end-to-end: samples for A and B, refusal scores for C, decoded through the shared decoder.
2. **kNN byte ledger reported against the store's**, with the shrink-to-match shown. ⛔ Not grown, not hand-solved.
3. Both LACE temperatures run; the whitening substitute named and justified; every concession declared.
4. Per-number provenance on every published constant: `PAPER:` / `OFFICIAL IMPLEMENTATION:` / `DERIVED:`.
   ⛔ Never inherit a library default and present it as the paper's value.
5. **Tuning-trial budget reported per arm**, equal across arms.
6. ⛔ **No kill-rule verdict.** You report arm numbers; `gen-cell-run` renders verdicts against a filed prereg.
7. Suite green (before/after, main venv); ref verified from the main repo before worktree removal.

## What would prove this spoke wrong
- ⭐ **You built a kNN arm that loses easily.** Two kill rules read off it; a weak one hands us a false pass. If your
  kNN looks bad, assume you hobbled it and try harder before you report it.
- You ran LACE unwhitened, or at the published temperature only. Either produces a straw competitor.

## ⟲ AMENDMENT 2026-09-09 — the whitening fallback is a declared arm, not a silent degradation
Head ruling: **our choices never weaken a competitor.** If the bridge's feature width does not admit full whitening,
LACE gets a **declared PCA projection and is reported both ways** — ⛔ never a silent drop to channel-only whitening.
Read the bridge's manifest for `D`, `N` and which option it took, and carry that choice through verbatim.
⚠ **C sets were regenerated at `|C| = |B|`** (Shapes3D **72**, dSprites **6,292**); the earlier 32-record files and
their checksums are superseded. ⛔ Read `c_invalid.json` from disk; never regenerate it.
⭐ **The primary refusal metric is AUROC(B vs C)** — kill rules (iii) and (iv) bind on it, and **(iv) reads off YOUR
kNN arm.** That is the sharpest reason not to hobble it.

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
