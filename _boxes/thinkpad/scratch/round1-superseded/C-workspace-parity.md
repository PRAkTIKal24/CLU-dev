# SPOKE C — Workspace parity: what is missing vs. a real hub  **(READ-ONLY)**

Read `.temp_cl/CONTEXT.md` first.
Report to: **`.temp_cl/reports/C-workspace-parity.md`**
Also read: `C:\Users\PratikJawahar\Downloads\WINDOWS-HUB-SETUP-BRIEF.md` — the brief that describes the
intended hub workspace. **It describes a bundle that is not on this machine.** Your job is to nail down
exactly what the gap is.

## The situation
The brief specifies a self-contained `chlu-hub-bundle.tar.gz` whose extracted root *is* `$CHLU_ROOT` and
contains:
```
chlu/ scripts/ tests/      <- present here, via git
.claude/                   <- charters, agent profiles, frozen artefacts (~1.95 GB)   [ABSENT]
env_hub.sh                 <- sets CHLU_ROOT and every derived CHLU_* path            [ABSENT]
BUNDLE_MANIFEST.json       <- every file + size + sha256                              [ABSENT]
README-HUB-BUNDLE.md                                                                  [ABSENT]
scripts/verify_hub_workspace.sh                                                       [ABSENT]
```
None of the absent items are tracked in git, and `.gitignore` excludes `.claude/**`, `docs/**` and
`projects/*/`, so **they can never arrive via `git pull`.** They must be shipped from the Mac.

## ⛔ You change nothing
Read-only. No installs, no writes outside `.temp_cl/`. Spoke A is installing Python in this directory
concurrently; ignore anything that appears under a venv path.

## Produce these, concretely

### 1. The exact manifest of what must come from the Mac
Do not guess from the brief alone — **derive it from the code**. Grep the repo for every `CHLU_*`
environment variable, every hard-coded path, every reference to `.claude/`, `projects/`, `docs/`,
`workspaces/`, charters, agent profiles, frozen artefacts, checkpoints, and datasets. For each, record:
- the variable/path,
- the file:line that reads it,
- whether it has a default or is required,
- whether it points at code (we have it) or data/state (we do not).
Output this as a **table**. This table is the actual deliverable — it is the shopping list the Mac side
will fill.

### 2. Reconstruct what `env_hub.sh` must have set
From the `CHLU_*` grep above, write out the **inferred contents** of `env_hub.sh` — every variable the
code expects and what it should be set to with `$CHLU_ROOT` = this repo root. Put it in your report **and**
in `.temp_cl/scratch/inferred_env_hub.sh` (⛔ not in the repo root). Flag every variable you had to guess.

### 3. Datasets and artefacts
The package has `chlu/data/` with loaders for dSprites, Shapes3D, MNIST, enwik8, wikitext, MQAR, and an
`industrial/` set (MIMII, SKAB, SMD-TSB, TEP-Rieth, Voraus-AD). For each loader, determine:
- where it expects data on disk (env var? relative path? download-on-demand?),
- rough size,
- whether it can self-download or must be copied from the Mac.
Anything under `chlu/eval/_tsb_vendor` is vendored and pinned — note it, do not touch it.

### 4. What state lives only on the Mac
Recent commits are tagged `[hub]` and `[experiment-engineer]` (e.g. `e4f3d71 [hub] merge
gen-store-grouping (gen-phase1)`). Read the git log (`git log --oneline -60`) and any in-repo docs/READMEs
to reconstruct: what is the current research phase, what is `gen-phase1`, and what running state
(experiment records, result caches, decision logs) would a hub agent on this machine need that is not in
git? Check `projects/` (contains only `README.md` here — read it), `logs/` (empty), and
`origin/internal-state` (a second remote branch — inspect it read-only with
`git log --oneline origin/internal-state -30` and `git ls-tree -r --name-only origin/internal-state | head -100`;
**do not check it out**).

### 5. Transfer mechanics
`rsync` is absent on this machine. Given the `.claude/` directory is ~1.95 GB and the repo sits on a
OneDrive Files-On-Demand path, recommend a transfer route Mac→Windows and flag the risks (OneDrive
rewriting/syncing 1.95 GB of frozen artefacts; checksum verification without `BUNDLE_MANIFEST.json`).

## Report shape
Lead with a one-paragraph verdict: **can this machine be a hub at all without the bundle, and if so for
which classes of work?** Then the tables. End with numbered open questions for the human.
