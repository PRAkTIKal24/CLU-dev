# w17 spoke/rendered-compositions — SHOW, DON'T ONLY TABULATE
**Read `COMMON.md` first.** Slug `rendered-compositions`. Agent type: results-analyst.

## The one job
The visual support: for a fixed set of held-out requests, render **the CLU's output beside EVERY rival's output on the SAME requests through the SAME frozen decoder**, with **the judge's verdict marked per panel**. Into `~/clu/outputs/w17-paper/figures/rendered/`.

## Request sets — ⭐⭐ THE HONESTY REQUIREMENT
Three panel sets, and **all three ship**:
1. One where **the CLU composes and rivals fail** (2-way and n-way corners, e.g. "red sphere, large, tilted" and full six-factor boxes).
2. ⭐ One where **ALL arms succeed**.
3. ⭐ One where **ALL arms fail**.
⛔ **Shipping only set 1 is cherry-picking and the figure would be dishonest.** All three, or say why one could not be built.

## Plus the deletion panel
The **same red-sphere request before and after deletion**: CLU vs rival-as-trained vs rival-retrained-without.

## Requirements
- Grid figures, **consistent panel labels**.
- ⛔ **No text inside panels** beyond the request label and the verdict mark.
- Same frozen decoder for every arm — **assert it in code**, don't assume it.
- Record the exact request tuples used, in JSON, so the figure is reproducible.

## ⚠ If the decoder or caches are not on this machine
The frozen decoder and the 28 GB caches live elsewhere. ⭐ **If you cannot reach them, write `NEEDS-ARTEFACTS.json` naming exactly which paths you need and say so in `LOG.md` immediately** — the Hub will stage them. ⛔ Do not stall silently, and ⛔ do not substitute a different decoder.

## Acceptance criterion
The three panel sets + the deletion panel as PDF+PNG, `REQUESTS.json` with the exact tuples, the same-decoder assertion shown, and any set you could not build DECLARED with its reason.

## Files you own
`figures/rendered/` only.
