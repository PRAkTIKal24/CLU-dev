# hepa-lh-sweep — the context curve, on the four datasets that can physically carry one

**ICLR long, wave 2. Agent: results-analyst. GPU via CSF3 (no local GPU).** NO worktree, NO branch, NO
commit — zero tracked files. Runs **concurrently with `hepa-lh-verdict`**; ownership disjoint (§5).
**Budget: ~2 days**, most of it queue time. Compute is not the constraint — the whole 14-dataset
5-seed sweep is under two hours on one A10G, and you are running four datasets.

## ⛔ READ FIRST, IN FULL
`.claude/outputs/hepa-longhorizon-triage/PREREG.md` (frozen; §1 protocol, §2 deltas, §3 rungs, §6 bans)
· `.claude/outputs/hepa-lh-audit.md` Tables A and B · `.claude/outputs/hepa-lh-tripwire.md` §2, §5, §6
· `.claude/AGENT_PROTOCOL.md` §2, §5.

## 1 · THE ONE JOB
Run **HEPA as shipped** at every reachable context rung on **four** datasets, 3 seeds, and produce the
**curve** — not the endpoint.

| dataset | rungs to run | ft_test events | note |
|---|---|---|---|
| ETTm1 | 512, 2048, 8192 | 92 | best powered of the 8192-capable |
| Beijing-AQ | 512, 2048 | 434 | best powered overall; **8192 is NOT REACHABLE** (val/test segments are 5 236) |
| PSM | 100 (shipped), 512, 2048, 8192 | 21 | shipped context is 100, not 512 |
| Weather | 512, 2048 | **7** | ⚠ **declared underpowered — run it, report it, and label every Weather number `UNDERPOWERED (n=7 test events)`.** It is included so the KEEP list is not a cherry-pick, not because 7 events can resolve anything. |

⛔ **Do not run** C-MAPSS FD001–4 (dropped on the audit: <1.1% of events have >512 steps of history),
MBA (ceiling **is** 512 — no rung above the reference exists), BATADAL (ceiling 100; 5 events total),
GECCO (ceiling 8192 but **one** ft_test event — the kill rule is not measurable on it). Also out of
scope: SMAP, TEP, VIX.
**Ceilings are `max_available_context` from `audit.json` and are hard.** A rung above the ceiling is
**NOT RUN**, reported as not-run, never as a null or a zero.

## 2 · ACCEPTANCE CRITERION (one, mechanical)
**`.claude/outputs/hepa-longhorizon-triage/sweep.json` exists**, containing for every (dataset, rung,
seed): h-AUROC, n_tokens at that rung, wall-clock, and the path to a **per-event test-prediction dump**
(`.claude/outputs/hepa-longhorizon-triage/preds/<dataset>_L<rung>_s<seed>.npz`). The dumps are not
optional — the `hepa-lh-verdict` spoke cannot do the information-age slice without them, and a rerun
to produce them costs another queue cycle.

## 3 · ⭐ THE REPRODUCTION — do this first, it is the most important run in wave 2
Wave 1's tripwire beat HEPA on FD001 by +0.060 — **but it was compared against a number read out of
HEPA's `REPRODUCIBILITY.md`, not against a run.** Until HEPA is run under our own harness on our own
cut points, that comparison is against a README and cannot go in a paper.

**So: run HEPA as shipped, at its shipped context, 3 seeds, on FD001 and GECCO**, and report your
h-AUROC against their stated 0.918±.008 (FD001) and .88±.06 (GECCO). FD001 is otherwise out of scope —
you are running it *only* to price the tripwire comparison, and it stays out of the triage table.
If your FD001 reproduction lands near 0.918, the tripwire's +0.060 stands and the benchmark has a
same-byte lookup beating it. **If it lands near 0.978, the tripwire result evaporates** and wave 1's
headline was an artifact of comparing against the wrong reference. Either way we need to know before
anything else is written. Report this before the sweep table.

## 4 · PROTOCOL — no drift
Clone-actual (`hepa/utils/config.py` + `REPRODUCIBILITY.md`), seeds **42, 123, 456**, patch **P=16 at
every rung** (so tokens scale with context — record `n_tokens` beside every h-AUROC; a longer context
that also buys more tokens is not a clean test of history and the report must say so). Horizon set
unchanged. **Scalar-concat horizon conditioning everywhere.** Per-dataset norm policy as shipped.
Use HEPA's own training and metric code from `.claude/hepa-src/` (@ `85f55f39`) — import, never
reimplement, never vendor (CC BY-NC-SA 4.0, `.claude/` only).
Report the **seed spread** (max − min) at the 512 rung per dataset — it is the denominator of the kill
rule and the verdict spoke consumes it.

## 5 · FILES YOU OWN
`.claude/outputs/hepa-longhorizon-triage/{sweep.json, preds/**}` · `.claude/outputs/hepa-lh-sweep.md` ·
`.claude/scratch/hepa-lh-sweep/**` · CSF3 job scripts under your scratch.
⛔ Not yours: `PREREG.md`, `audit.json`, `PROVENANCE.md`, `code/**` (read-only — invoke the tripwire
harness, do not edit it), `verdicts.json`, any tracked file. `.claude/data/hepa/**` is **read-only**.

## 6 · WHAT WOULD PROVE THE IDEA WRONG
The idea is that some dataset here needs history older than 512 steps. **You falsify it by returning
flat curves** — h-AUROC at 2048/8192 within the 512 seed spread on all four. Given the audit found the
most recent analogous regime sits 46–193 steps back on every one of these datasets, **flat is the
outcome to expect**, and a flat curve is the result, not a failed run. Report it in your first line.

## 7 · REPORT
`.claude/outputs/hepa-lh-sweep.md`, protocol §5. The reproduction first, then one curve per dataset
(h-AUROC ± spread vs rung, with n_tokens), then the not-run list with reasons. Every number traces to
a path. ⛔ **No CLU arm, no CLU number.** Under ~2 pages.
