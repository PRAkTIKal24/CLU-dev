# Task: gen-feature-bridge — the frozen encoder/decoder bridge, and the ceiling that gates the whole phase

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-feature-bridge.md` + artefacts under `.claude/outputs/gen-feature-bridge/`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (all of it) · **`.claude/tasks/gen-clu-phase1-rulings.md` — R1, R3, R5, R6 are binding on you** · `.claude/outputs/gen-data-splits.md` (your input) · `chlu/data/gen_splits.py`. ⛔ Read no campaign ledger, charter or handover — different programme, it will mislead you.
- **Base:** `main`, **after `gen-data-splits` has merged** · **Branch:** `agent/experiment-engineer/gen-feature-bridge` · **Worktree:** yes, fresh (`../CHLU-gen-bridge`). ⛔ `CHLU-bprime-family2`, `CHLU-wt2`, `CHLU-wt3`, `CHLU-merge` are parked, not free.
- ⛔ **PRECONDITION — mechanical.** Do not start until **both** hold:
  `git cat-file -e main:chlu/data/gen_splits.py` succeeds, **and**
  `.claude/data/gen/shapes3d/c_invalid.json` exists.
  If either fails, **stop and report.** Do not build your own loader or split.

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** *none — instrument, plus one gating measurement.* ⛔ No store, no wells, no arm, no CLU number, no
  composition claim. Nothing here is evidence for the primitive.
- **Laundering control:** the A-split equivalent of the same ceiling measurement — the ceiling is meaningless in
  absolute terms and is only ever read as a ratio to it.
- **Falsifies:** `ceiling_B < 0.8 x ceiling_A` **stops the phase** (R3.1). That is a legitimate, useful outcome.
- **Does NOT falsify:** a low absolute probe accuracy that is low on **both** A and B. That is a weak probe or a hard
  dataset, not a broken bridge, and the ratio is what is under test.

## The one job
Build the frozen encoder, the cached penultimate features, and the **one frozen decoder that every arm shares
forever** — and return the number that says whether the phase may proceed.

## Why this is a gate and not a control (R3)
The decoder is trained on train-split pairs only, so it may simply be unable to render a combination it never saw.
If so, B accuracy is low for **every** arm, store-removed matches CLU, and **kill rule (i) fires on the decoder rather
than on the store** — we would spend the phase's compute proving something about a decoder and read it as a result
about the primitive. ⭐ **So the ceiling is measured before any arm exists, and it can stop the phase.**

## Head rulings that govern this task — quoted, because they are easy to over-apply
- **R1 — reconstruction is the mandatory primary pretext.** `cls` is built as a **diagnostic arm only, never a headline
  row.** ⭐ **CLARIFICATION: the evaluation probe classifier still trains on factor labels. That is measurement, not the
  generative path, and it stays.** ⛔ Do not strip label supervision from the probe in the name of R1 — that would
  destroy the measurement instrument, not protect the claim.
- **R3.3 — no leak.** The decoder **never** trains on held-out-combination images. They are used at evaluation only.
  **Assert it and report the count of excluded images.** Later "just fix the decoder" pressure points at exactly this
  leak, so the assertion is what protects the result from us.

## Files you own (nobody else touches these)
- `chlu/core/gen_bridge.py` · `chlu/experiments/exp_gen_bridge.py` · `chlu/eval/gen_probe.py` — all new.
- `chlu/config.py` — **shared, minimal hunk.** Append **one** dataclass (`GenBridgeConfig`) and **one** field on
  `CHLUConfig`, *after* the hunk `gen-data-splits` already landed. ⛔ Nothing else; report exact line ranges.
- `chlu/cli/experiment_cmd.py` — **append one subparser only.** ⛔ Touch no existing line.
- `tests/test_gen_bridge.py` — new.
⛔ You own **no** file under `scripts/csf3/` — `job_gpu_gen.sh` belongs to `gen-csf3-launcher`. Use it; do not edit it.

## What to build
1. **Encoder**, small conv, trained on the **train split only**, pretext **reconstruction** (R1). Declare the pretext in
   the artefact manifest. Then **freeze it** and record its parameter hash.
2. **Cached penultimate feature maps for the full dataset** (train + held-out + everything the factor grid indexes),
   written once, with a checksum and a manifest. Everything downstream reads this cache; it is never recomputed per arm.
3. **Decoder**, conv, penultimate-feature space → pixels, trained on real `(feature, image)` pairs from the **train split
   only**, then **frozen**. ⭐ **One decoder, shared by every arm, forever.** Record its parameter hash and emit it where
   downstream spokes can assert against it — an arm that silently retrains the decoder invalidates the whole cell.
4. **Probe classifier**, per-factor, trained on **real** images with factor labels (the R1 exemption). This is the
   measurement instrument for the whole phase. Report its accuracy on real held-out images — ⭐ **a probe that cannot
   read real images cannot adjudicate generated ones**, and if it is weak, say so before anything is built on it.
5. **The ceiling measurement.** Feed the **real cached features** of held-out-combination images through the frozen
   decoder; probe the decoded images; report per-factor and aggregate accuracy as `ceiling_B`. Repeat on the seen
   split for `ceiling_A`, same probe, same protocol.
6. **The `cls` diagnostic arm:** the same pipeline with a factor-classification pretext, reported in a clearly
   separated diagnostic section. ⛔ Never in the same table as a `recon` number.

## Acceptance
1. Frozen encoder, feature cache, frozen decoder and probe all exist, each with a parameter/content hash and a manifest.
2. **`ceiling_A` and `ceiling_B` reported, per-factor and aggregate, ≥3 seeds, with spread.**
3. ⭐ **The gate verdict, stated explicitly:** `PROCEED` if `ceiling_B >= 0.8 x ceiling_A`, else **`STOP — return to Head
   and Advisor`**. ⛔ A `STOP` is a legitimate result. Do not tune the decoder to clear the floor and do not soften the
   verdict; if it stops, it stops.
4. The no-leak assertion runs and its excluded-image count is reported.
5. Probe accuracy on **real** held-out images reported, so the instrument is auditable.
6. Runs stay inside the budget: ≤30 min per dataset × seed on one GPU, via `scripts/csf3/job_gpu_gen.sh` with
   `STAGE=encoder` then `STAGE=decoder`. Local CPU smoke first (R5). Paste job ids and realised paths.
7. Suite green against your base. Record the pass count before and after using the **main venv** per protocol §4
   (`PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python -m pytest -q`). ⚠ A fresh worktree venv drops ~7
   tests via optional extras — not a regression, but say so rather than reporting a smaller number.
8. Re-verify `git log --oneline main..agent/experiment-engineer/gen-feature-bridge` **from the main repo** before
   removing the worktree.

## What would prove this spoke wrong
- **`ceiling_B` is high because the decoder saw held-out combinations.** The R3.3 assertion is what catches it. Run it
  and report the count before you believe your own ceiling.
- **The probe is weak on real images**, in which case every accuracy in this phase is measuring the probe. Check item 5
  before reporting item 2.
- You reported an absolute `ceiling_B` without `ceiling_A`. The number means nothing on its own.

---

# ⛔ CONSTRAINT folded in from `gen-lace-spec` on 2026-09-09, BEFORE this spoke spawned

## The penultimate feature dimension is now a design constraint, not a free choice
The LACE baseline arm is only valid in **whitened** coordinates: its `½‖z‖²` term is the exact log-density of
StyleGAN's z-space, and that exactness is what licenses its ODE drift to omit the prior score. On raw post-ReLU
features the shipped drift is **the wrong vector field**, so the arm must fit a shrinkage Gaussian over the cached
train features and sample in `u = Σ^{-1/2}(x − μ)`. That requires a `D × D` eigendecomposition over the flattened
penultimate dimension `D = C·H·W`.

⭐ **Therefore: choose the encoder so that `D = C·H·W` stays at or below ~2e4.** Above roughly that, full whitening
becomes infeasible and the LACE arm falls back to channel-only whitening (`C × C` over space and batch) — a declared
deviation that weakens the strongest competitor arm. ⛔ **We do not want to hobble LACE by an architecture choice made
for our own convenience.**
- **Report `D` explicitly in your manifest**, with a one-line statement of whether full whitening is feasible at that `D`.
- If your best reconstruction encoder needs `D > 2e4`, **say so and report it as a flagged trade-off** rather than
  silently picking the smaller encoder or silently pushing LACE onto channel whitening. It is a Head call.
- Also record `N` (number of cached train features): shrinkage is **mandatory** where `D > N`.

## Consequence for the shared decoder (resolves the LACE scout's reconciliation item R1)
The scout flagged a Hub choice between feature-space and image-space scoring, noting LACE cannot render from conv
features without a decoder. ⭐ **Resolved by the design, not escalated: scoring is image-space, and every arm — CLU,
store-removed, LACE, kNN — decodes through the ONE shared frozen decoder you build here.** LACE is therefore not
generator-free in our setup, but it uses the identical decoder as every other arm, so no arm gains or loses by it.
⛔ No arm may substitute its own decoder. Emit the decoder's parameter hash so downstream arms can assert against it.

---

# ⛔ AMENDMENT 2026-09-09 (before spawn) — reading exemption, and what landed upstream

## Reading exemption (corrects this file's "Read first" line)
The ⛔ *"read no campaign ledger, charter or handover"* stands, **with one exemption: you MUST read
`.claude/handover_context.md` §7 Known Issues.** Without it a spoke can silently re-introduce a bug this programme
already fixed. Read §7 only — not the running log, not the charters.

## What is already on `main @ 9e87f72` — use it, do not rebuild it
`gen-data-splits` has merged. Your preconditions are **MET**. Available to you:
- `chlu/data/gen_splits.py` — `FactorGrid` (row↔factor bijection, exhaustively verified on both datasets),
  `GenSplit`, `HoldoutViolation`, the three assertions, the C-set builder/validator.
- `chlu/data/shapes3d.py` / `chlu/data/dsprites.py` — loaders with checksum verification and **verbatim** Montero
  filters. ⛔ Do not write your own loader or re-derive a split.
- `chlu data shapes3d [--download] [--quick]`, `chlu data dsprites` · config block `gen_data` (`GenDataConfig`) ·
  env var **`CHLU_GEN_DATA_ROOT`** (⚠ distinct from the industrial `CHLU_DATA_ROOT`; **worktrees have no `.claude/`,
  so you must set it**).
- **Realised held-out counts: Shapes3D 72/480,000 · dSprites 6,292/737,280.** ⛔ Not 5,808 — that nominal figure is
  retired. Assert against these.

## ⚠ h5py is an undeclared dependency (R12)
Shapes3D is a single HDF5 file. h5py is present in the main venv but **not** in `pyproject.toml` or
`requirements-csf3.txt`, so a `uv sync` prunes it and the real-data tests silently become skips.
⛔ **Do not add it to `pyproject.toml` or re-lock — that is a scheduled Hub job and would break concurrent spokes.**
Install it manually where you need it (`uv pip install h5py`, and on CSF3 into the job venv) and **report that you did**.

## ⚠ Do not materialise the images
`open_images()` returns a lazy h5py handle. ⛔ **Never `f['images'][:]`** — that is 1.7 GB. Stream or index in chunks.
Nothing in the code enforces this; it is on you.

## ⟲ SUPERSEDED 2026-09-09 — h5py is now declared
The h5py note above is **retired**. `main @ 40f606d` declares it as a `gen` extra: **`uv sync --extra gen`**, and
`scripts/csf3/requirements-csf3.txt` is regenerated with it. ⛔ Do **not** hand-install it and do **not** re-lock.
Your base is now `main @ 40f606d` (the launcher and the h5py extra have both landed).

## ⛔ CLI CONTRACT — binding, from `gen-csf3-launcher`
Every gen stage module **must** accept exactly these flags or `job_gpu_gen.sh` aborts on it:
`--dataset --pretext --stage --arm --seed --out`. Your `exp_gen_bridge` entry point implements them.
Submit lines carry `GEN_GIT_COMMIT=$(git rev-parse HEAD)` — the launcher refuses to run without it. The Hub writes
the submit lines; you just honour the contract.

## ⟲ SUPERSEDED 2026-09-09 (second time) — the LACE width question is RULED, not escalated
The "if `D > 2e4`, report it as a flagged trade-off, it is a Head call" wording above is **retired**.
⭐ **Head ruling: our choices never weaken a competitor.** So you do one of two things, and you decide it:
1. **Pick a feature width LACE can whiten natively** (full `D × D` shrinkage covariance feasible), **or**
2. **Keep the wider encoder and hand LACE a declared PCA projection**, with the arm **reported both ways**.
⛔ Do not silently pick the narrow encoder to make our own life easy, and ⛔ do not push LACE onto channel-only
whitening without declaring it. Record `D`, `N`, and which option you took, in the manifest.
**Base is now `main @ 215fc4d`.** ⚠ The C sets were regenerated at `|C| = |B|` (Shapes3D 72, dSprites 6,292) —
⛔ the 32-record files and their checksums are superseded.
