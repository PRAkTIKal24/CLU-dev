# w15 spoke/bar-d-score — CLOSE BAR D
**Read `COMMON.md` first.** Slug `bar-d-score`. Agent type: results-analyst.
**Scored: `encoder-loop` 1 run. ⭐ NEEDS A NEW PREREG.**

## The one job
Score the **finished GPU leg (full decoder)** under **its own budget pins** and close BAR D.

⛔⛔ **A NEW PREREG IS MANDATORY.** The GPU pins are **4000/32768**; the CPU leg's are **400/8192**. The code's own rule is that different budgets do not share a table. ⛔ **Never merge this with the w8 CPU table.**

## Mechanics
- Scorer: `scripts/w8_encoder_loop_score.sh` + `w8_encoder_loop_report.py`.
- ⛔ `test` refuses without a PREREG **and** a row in `/c/chlu/SCORED-LEDGER.md`. Write both first.
- ⛔ **Do not fabricate a ledger on the Mac.** The ledger is `/c/chlu/SCORED-LEDGER.md` on this box.
- ⚠ **The arms and caches are on the MAC, not this box** (`.claude/scratch/w8-gpu/out`, 28 GB). The arms also remain in `gs://chlu-cloud-forgisprova/jobs/w8enc-s{0,1,2}/`. Caches regenerate in ~162 s/arm. ⭐ **If the inputs are not reachable from this box, say so immediately in `LOG.md` and write a `NEEDS-MAC.json` naming exactly which paths you need** — the Hub will stage them. ⛔ Do not stall silently.
- ⚠ **The handover claims 46 manifests at `outputs/w8-encoder-loop/cloud/manifests/`; the Mac shows far fewer.** Count what actually exists and report the real number. ⛔ Do not cite 46 unless you counted 46.

## The paragraph this feeds
The **decomposition paragraph**: *unfreezing hurts / swap loss repairs*. Your artefact is what it cites.

## Acceptance criterion
`BAR-D.json` + `FINDINGS.md`: the scored row under the new PREREG with its budget pins named in the row itself, the ledger row written before the run, the manifest count as measured, and an explicit statement that this never shares a table with the w8 CPU leg.

## What would prove this wrong
If the GPU leg does not reproduce the CPU leg's ordering, that is the finding — print both with their pins and decline to merge them. ⭐ A negative here is a clean result.

## Files you own
Your worktree, your output dir, your ledger row, and a new `PREREG-w15-bar-d.md`.
