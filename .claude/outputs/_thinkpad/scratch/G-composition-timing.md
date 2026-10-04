# SPOKE G — Replace the `exp_gen_cell` timing guess with a measured number
**Core budget: `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, ONE unit at a time.** Two other spokes share
these 16 cores. ⛔ Do not let JAX or BLAS take all 16 — an unbound numpy/sklearn oversubscribes and runs
**slower**, not faster.

Read `_hub_scratch/CONTEXT.md` first.
Work in: `/c/Users/PratikJawahar/chlu/chlu-hub-bundle`
Report to: **`/c/Users/PratikJawahar/chlu/_hub_scratch/reports/G-composition-timing.md`**

## Why you exist
The Head has ruled that **CPU jobs run locally on this box**, and `hub/composition` is the recommended
first hub here — every input it needs is already in the bundle. But **nobody has ever run `exp_gen_cell`
on this machine, and there is no timing for it anywhere.** The current estimate, "~1–3 h/seed on 16
cores", is an inference anchored on `exp_gen_baselines`' measured ~0.85 h/seed. That is an order of
magnitude, not a budget, and a ten-hour autonomous run is about to be planned against it.

**Your job: one cheap, explicitly non-scored probe that turns that guess into a number.**

## ⛔⛔ THE LINE YOU DO NOT CROSS — this is a scientific-integrity boundary, not a style rule
The charter caps a hub at **3 scored runs against the held-out sets**, for the whole hub. A scored run you
burn here is one the real hub cannot have, **and it cannot see that you spent it.**

- ⛔ **Do NOT perform a scored run.** ⛔ Do not touch the held-out query sets. ⛔ Do not run the aggregate
  stage. ⛔ Do not produce, and do not write down, any number that could be quoted as a result.
- ⭐ **You are measuring wall-clock, memory and whether it runs at all.** Nothing else.
- Use `--quick` (or whatever the CLI's equivalent reduced mode is — **read the flags first**), one seed,
  one arm. Prefer the `clu` arm, or a cheaper one if the code makes that obvious.
- ⛔ **Do not start `hub/refusal` or any part of it.** Its artefacts are missing from this bundle; a
  supplement tarball is being built. See `CONTEXT.md`.
- ⛔ `.claude/outputs/gen-arms-a7-wiring/refusal_scores.json` is the SUPERSEDED broken-split file: **never
  quote it**, and never move or delete it — it is sha256-pinned.

## Write surface
`$CHLU_HUB_OUTPUTS/<slug>/**` only, and use a **throwaway probe slug** (e.g. `probe-timing`), ⛔ never
`hub-composition`. Plus your report. The charter's **§0.1 YOU NEVER DELETE** applies to you: leave what you
write in place, and list it in your report.

## Steps
1. **Read before you run.** `chlu/experiments/exp_gen_cell.py` — its `ARMS`
   (`clu, store_removed, knn_proto, lace, ebm_compose, dynread`), its `--stage` values (`cell`,
   `aggregate`), its seed/temperature flags, and the pre-registered temperature grid
   (`temp ∈ {0.0, 0.005, 0.05, 0.5}`). Report the **exact CLI surface** — the hub will need it, and no
   accurate write-up of it exists yet.
2. **Confirm the inputs resolve.** Every `CHLU_*` root from `env_hub.sh` should point at something real.
   ⚠ Two are expected to be **absent**: `CHLU_GEN_WIDE_BRIDGE_ROOT` (`gen-bridge-wide/runs`) and anything
   under `gen-read-cell-arm/`. Confirm which roots resolve and which do not, and whether `exp_gen_cell`
   needs any of the missing ones. **If it does, stop and report — that is a bigger finding than a timing.**
3. **One `--quick` cell probe**, one seed, one arm, with the thread pinning above. Time it with real
   wall-clock and sample peak RSS (`Get-Process`, or `resource` is unavailable so use PowerShell /
   `psutil` if present). Capture full stdout/stderr to
   `_hub_scratch/reports/_G-probe.log`.
4. **Extrapolate honestly.** From the `--quick` unit, estimate a full unit and then the full grid
   (6 arms × 3 seeds × 4 temps = **72 units**). ⭐ **State your scaling assumption explicitly and flag it
   as an assumption** — if `--quick` reduces steps by 10× that is not the same as 10× the time.
5. **Memory headroom.** The three feature caches are 1.83 GiB each (float16, 480,000 × 128×4×4), read via
   `mmap_mode="r"`. Confirm the probe's RSS stays modest and that mmap behaves on NTFS — on a 16 GB Mac
   this contention is exactly what "BLOCKED the scored cell arm"; on 63 GB all three seeds should be page
   cache. **Report whether that holds in practice**, because it is the core argument for this box.
6. **Concurrency sizing.** Given your measured single-unit cost and RSS, recommend how many units to run
   in parallel and at what `OMP_NUM_THREADS` each (e.g. 4 units × 4 threads). This directly informs
   whether the 3-sub-agent cap should be raised, which the Head has said is open on this box.

## What NOT to spend time on
⛔ Do not tune, optimise or improve anything. ⛔ Do not run the other five arms. ⛔ Do not run more than
one seed. ⛔ Do not attempt the full grid. If the probe reveals the code is broken on Windows, **report it
and stop** — that is the most valuable outcome available to you and it changes the whole plan.

## Report shape
Lead with three numbers and one verdict: **seconds for the probe unit · extrapolated hours for one full
seed · extrapolated hours for the 72-unit grid · does `hub/composition` fit in a ten-hour run on this box,
yes or no?** Flag every extrapolation as an extrapolation. Then the CLI surface, the input-resolution
table, memory, and the concurrency recommendation. End with numbered open questions.

⭐ Before you finish, **confirm no sub-agent of yours is still running**, and list every file you left
behind under `$CHLU_HUB_OUTPUTS`.
