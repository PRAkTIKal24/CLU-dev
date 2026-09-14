# hepa-lh-tripwire — build the thing that can kill a KEEP, before any KEEP exists

**ICLR long. Agent: experiment-engineer. CPU only, local. NO worktree, NO branch, NO commit** — you
touch zero tracked files. All code you write lives under `.claude/`.
Runs **concurrently with `hepa-lh-audit`**; ownership is disjoint (§5). **Budget: ~1.5 days.**

## ⛔ READ FIRST, IN FULL
1. `.claude/outputs/hepa-longhorizon-triage/PREREG.md` — **frozen, you may not edit it.** §1 names the
   protocol file, §2 the permitted deltas, §4 the kill rule you are building the instrument for.
2. `.claude/outputs/hepa-benchmark-scout.md` §4 (label construction), §6 (horizon conditioning), §7
   (encoder/predictor/head shapes and parameter counts).
3. `.claude/AGENT_PROTOCOL.md` §2, §5.

## 1 · THE ONE JOB
Build the **matched-byte tripwire**: the cheap reference that decides whether a long-context h-AUROC
gain is real or just more bytes. Concretely, a **kNN exemplar store over window features at HEPA's own
state-byte budget**, plus **the best trivial baseline** (prevalence / seasonal / last-event-age),
scoring **h-AUROC on HEPA's exact horizon grid**.

You are building the instrument that can kill the result. **It ships before the result exists, on
purpose.** A tripwire authored after seeing a sweep is worthless.

## 2 · ACCEPTANCE CRITERION (one, mechanical)
**`.claude/outputs/hepa-longhorizon-triage/code/tripwire.py` runs end-to-end on C-MAPSS FD001 and
writes `.claude/outputs/hepa-longhorizon-triage/tripwire_smoke.json`** containing, at context 512 and
at 3 seeds: the exemplar-store h-AUROC, every trivial-baseline h-AUROC, the computed state-byte
budget, and the realised store size in bytes.
"Runs end-to-end" means: reads via HEPA's own loader, produces h-AUROC via HEPA's own metric code.

## 3 · MECHANICAL PRECONDITION — verify on disk before you validate anything
> **`.claude/data/hepa/cmapss/READY`** must exist.
It is written by the `hepa-lh-audit` spoke once C-MAPSS FD001 is staged.
**Write your code immediately — do not wait.** Only §2's validation run is gated. If `READY` has not
appeared by the time your harness is finished, report `PARTIAL`, hand over the unvalidated code, and
say exactly that. Do not substitute synthetic data for the validation and call it done.

## 4 · WHAT "MATCHED BYTES" MEANS — get this right or nothing downstream is interpretable
The exemplar store must be sized to the bytes **HEPA itself carries at that context**, so that any win
is a win of *mechanism*, not of *capacity*.
1. Compute HEPA's per-example state footprint at context L from the shipped encoder config
   (256-d, P=16 so L/16 tokens, layer count and dtype from `hepa/model/encoder.py`). Show the
   arithmetic in a **byte ledger** in your report — the Hub reviews the ledger before wave 2 uses it,
   and a wrong ledger silently invalidates every KEEP verdict in this task.
2. Size the kNN store to that same byte count: n_exemplars x feature_dim x dtype <= budget. When the
   budget forces a subsample, **the seed controls the subsample.**
3. Recompute the budget **at every context rung** — it grows with L, and the store must grow with it.
   A store fixed at the 512-rung size makes the 8192 rung a strawman.

**Horizon conditioning (PREREG §2.4, binding):** every arm you score uses HEPA's **scalar-concat**
mechanism — raw dt in native units. Do **not** give the tripwire the 264K learned horizon embedding
the published baselines use. One conditioning mechanism across everything compared, or it is not a
comparison.

**h-AUROC:** import HEPA's implementation. Do not write your own — per-horizon prevalence in this
suite runs from 0.5% to 96%, and a hand-rolled metric that differs in how it drops degenerate horizons
will disagree with every published number for reasons no one will find later. Note that the paper
thresholds degenerate horizons at 0.001/0.999 while the code tests exactly-0-or-1 (scout D4); use the
code's, and say so.

## 5 · SEEDS, HONESTLY
PREREG fixes seeds **42, 123, 456**. For a kNN store the seed controls the exemplar subsample and
tie-breaking. **If at some budget the store is exactly deterministic** (no subsampling needed, no
ties), then three seeds give three identical numbers — in that case report **one number explicitly
labelled `deterministic`**, not a fake +/-0.00 across three runs. Say which regime each dataset is in.
Fabricating spread is worse than having none.

## 6 · TRIVIAL BASELINES — all three, each reported
Report every reference you compute; the strongest is the one the kill rule uses, but a suppressed weak
baseline is a suppressed result.
- **prevalence**: per-horizon base rate from train.
- **seasonal**: period from the dataset's native rate (daily/weekly where meaningful); state the period.
- **last-event-age**: rank by time since the entity's previous event — the single most likely thing to
  beat a neural model on this suite, and the one a referee will ask for by name.

## 7 · FILES YOU OWN (nothing else)
- `.claude/outputs/hepa-longhorizon-triage/code/**` — the harness. **Not `.claude/scratch/`**: scratch
  is regenerable and excluded from backups, and this harness is neither.
- `.claude/outputs/hepa-longhorizon-triage/tripwire_smoke.json`
- `.claude/outputs/hepa-lh-tripwire.md` — your report.
⛔ You do NOT own: `PREREG.md`, `audit.json`, `PROVENANCE.md`, `.claude/data/hepa/**` (read-only to
you — read the staged data, never write or re-download it), any tracked file.
Reuse the audit spoke's clone at `.claude/hepa-src/` read-only; do not clone your own copy.

## 8 · WHAT WOULD PROVE THE IDEA WRONG
The idea is that HEPA's encoder does something a same-byte lookup cannot. **You falsify it by building
a tripwire that matches or beats HEPA at 512 on FD001** — if a kNN exemplar store at equal bytes, or
last-event-age with no model at all, ties the shipped number, then the suite's margins are not
evidence of representation learning and the long-horizon section has a much bigger problem than
context length. That outcome is a finding. Report it in the first line of your report if it happens.

## 9 · REPORT
`.claude/outputs/hepa-lh-tripwire.md` per AGENT_PROTOCOL §5. Must contain: the byte ledger with its
arithmetic, the smoke numbers, the degenerate-horizon convention you used, which datasets will be
deterministic-store, and the exact call signature wave 2 uses to invoke your harness at an arbitrary
context rung. ⛔ No CLU arm, no CLU number. Under ~2 pages.
