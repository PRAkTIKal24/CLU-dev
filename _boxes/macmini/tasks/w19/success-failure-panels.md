# w19 spoke/success-failure-panels — SHOW EVERY ARM SUCCEEDING AND FAILING
**Agent type: results-analyst.** Zero scored runs. ⭐ PRE-AUTHORISED; never wait for anyone.

## The one job
Rendered outputs, **judge-marked**, for **BOTH** tasks, **per arm**.

### Shapes — extend the w17 panels
Keep w17's **set 1** ("CLU composes, rivals fail") and **set 3** ("all fail") as they are.
**Add:** for the CLU **and for EACH rival**, **one row where it succeeds and one where it fails** on the **same held-out requests**.
⭐⭐ **Add a "rival succeeds, CLU fails" row if any such request exists on any seed — and if none exists, DECLARE that.** ⛔ That row existing is not a problem; its absence being unstated would be.

### CMAZE-T
For a fixed set of **held-out n-way transitions** drawn from the **five named families**: draw **s_t + action**, **the engine's true s_{t+1}**, and **each arm's generated s_{t+1}**, with **legal/illegal marked**.
**One success and one failure per arm.** Plus **one panel set on the door-then-tint interaction specifically**.
**Post-deletion panel:** the same transition **before and after deleting colour c**, for the **CLU**, **rival-as-trained**, and **rival-retrained-without**.

## Requirements
- ⛔ **From artefacts only.** ⭐ **Same frozen decoder/codec, hash-asserted** — assert it in code, never assume it.
- ⭐ **Byte-identical on rebuild.** Build twice, `shasum`, show the match.
- ⛔ **No in-panel text** beyond the request/words label and the verdict mark.
- Consistent panel labels and arm names with every other figure.
- Record the exact request tuples / transitions used, in JSON, so the figure is reproducible.

## ⚠ If an artefact is not on this machine
⭐ **Write `NEEDS-ARTEFACTS.json` naming exactly which paths you need, say so in `LOG.md`, and continue with what you CAN build.** ⛔ Do not stall silently and ⛔ do not substitute a different decoder. (The w17 `rendered-compositions` spoke did exactly this and it worked — its request is a good template.)

## Acceptance criterion
All panel sets as PDF+PNG in `~/clu/outputs/w17-paper/figures/rendered/`, `REQUESTS-w19.json` with the exact tuples/transitions, the same-decoder assertion shown, the rebuild proof, and **any set you could not build DECLARED with its reason**.

## What would prove this wrong
⭐ If **no** held-out request exists where a rival succeeds and the CLU fails, that is a real asymmetry worth stating. ⭐ If **many** do, that is equally worth stating. ⛔ Either way it is reported, not selected around.

## Files you own
`figures/rendered/` and `REQUESTS-w19.json` only.
