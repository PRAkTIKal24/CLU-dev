# Task: gen-loss-attribution — WHERE does `scale` get lost? Encoder, well, or code?

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-loss-attribution.md` + artefacts under `.claude/outputs/gen-loss-attribution/`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (all) · **`.claude/tasks/gen-clu-phase1-rulings.md` — R40 (A.2, and ⛔ B in full: the compose path is a product quantiser + count table), R31 (the corrected mechanism + the `n_values_realised` rule), R34 (the loss is GLOBAL, not corner-local)** · `.claude/outputs/gen-commitment-collapse.md` · `.claude/outputs/gen-navigation.md` §5.
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Base:** `main @ 994ca9b` (or the Hub's tip at spawn) · **Branch:** `agent/experiment-engineer/gen-loss-attribution` · **Worktree:** fresh (`../CHLU-gen-attrib`).
- **Env:** main venv. ⭐ **Run pytest with `CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen`** — a worktree has no `.claude/`, and without it real-data tests **silently skip**. **Report passed / skipped / failed, never passed alone.**
- ⚠ **Budget ≈ 3 h.** Local, 3 seeds. ⭐ Write per seed, skip any seed whose artefact exists.

## ⭐ DIAL DECLARATION
- **Dial:** *attribution* — ⛔ **no fix, no new mechanism, no arm.** You are locating a known loss, not repairing it.
- **Laundering control:** the **same probe, same protocol, same rows** at all three points. ⛔ A different probe at each
  stage would measure the probes, not the pipeline.
- **Falsifies:** nothing of ours — ⭐ **this spoke is designed so that BOTH outcomes are informative**, and which one
  it returns decides where the next wave's effort goes.

## The one job
Probe **`scale`** and **`orientation`** at **three points on the same rows** and say where the information disappears:
1. **frozen-encoder features** (the 2048-d cache),
2. **the per-well sub-vector** (the slice of those features assigned to each well),
3. **the store code** (the integer K-tuple after k-means + argmax).

## ⭐ THE PRE-REGISTERED READING — file it BEFORE measuring, and do not edit it after
| outcome | reading |
|---|---|
| encoder **≥ ~0.9** AND code **≈ chance** | ⭐ **World A — the STORE drops it.** The fix is in the write-side quantiser. |
| encoder **lacks it** | ⭐ **World B — the store is INNOCENT.** The fix is the encoder, and the encoder track owns it. |
⛔ Anything in between is reported as in-between, ⛔ never rounded to whichever world is more convenient.
⭐ **Report the per-well sub-vector point too** — it separates *"the partition threw the factor away"* from *"k-means
threw it away"*, which are different repairs.

## ⭐ THE FREE INSTRUMENT — use it, it costs nothing
**The mutual information between each well's codes and each factor is ALREADY COMPUTED in the store.** Read it out
and tabulate it. ⭐ **A flat `scale` row is the collapse seen from the address book's own side** — an independent
view of the same fact, from data already on disk. ⛔ If it disagrees with the probe, report the disagreement; do not
pick one.

## ⛔ WHAT THE PATH ACTUALLY IS — read this or you will mis-describe your own measurement
The compose path is a **product quantiser + count table**: correlation-partitioned wells × per-well **k-means**
codebook; the address book is **per-factor-value mode histograms** combined naive-Bayes and **argmax'd per well**;
the read is a **codebook copy-back**. ⛔ **There is no landscape, no particle, no settling and no inter-well coupling
on this path.** ⭐ The working hypothesis for the loss is therefore mundane and testable: **k-means keeps
high-variance directions**, and `scale`/`orientation` are low-variance in these features.
⛔ **Do not write "k-particle launch read" anywhere** — it names an argmax over a count table.

## ⛔ Live traps
1. **The same probe at all three points.** Same class, same regularisation, same train/test rows, same seeds. ⛔ Report
   the probe's own capacity — a linear probe on a 30-bit code and on a 2048-d vector are not equally expressive, and
   ⭐ **that asymmetry must be stated, because it could by itself produce a "loss" that is not there.**
   ⭐ **Mandatory control: probe a RANDOM code of the same width.** It bounds what the code-level probe can score
   with no information at all.
2. ⚠ **A per-factor statistic is meaningless without `n_values_realised`.** Montero's corner pins `shape` to ONE
   value and yields a confident-looking zero. **Report `n_values_realised` beside every per-factor number.**
3. ⭐ **Measure on the SPREAD grid, not only the corner.** The loss is **global** (`scale` 0.229 vs chance 0.125 on
   the grid), so a corner-only measurement will overstate it.
4. ⚠ **Feature caches are non-uniform:** seed 0 is `scope="full"`; seeds 1–2 are `scope="used"` and contain **none**
   of the matched-seen rows.
5. **JAX cold start ~20 min. 16 GB machine** — ⛔ never start a full suite.

## Files you own
`chlu/eval/gen_attribution.py` (new) · `chlu/experiments/exp_gen_attribution.py` (new) ·
`tests/test_gen_attribution.py` (new) · `chlu/config.py` — **ONE new block, APPENDED AT EOF.**
⛔ **Read-only on everything else** — `gen_store.py`, `gen_compose.py`, `gen_dream.py`, `gen_bridge.py`,
`chlu/eval/rivals/`, `chlu/data/`, `scripts/csf3/`. ⛔ **You fix nothing in this spoke.**

## Acceptance
1. `PREREG.md` filed **before** the first measurement, containing the World A / World B table verbatim.
2. `scale` and `orientation` probed at all three points, same probe, 3 seeds, on the **spread grid** and the corner
   **reported separately**.
3. The random-code control reported beside the code-level number.
4. The store's existing per-well MI tabulated, and any disagreement with the probe stated, not resolved.
5. `n_values_realised` beside every per-factor number.
6. **A verdict: World A, World B, or explicitly in-between** — one line, in those words.
7. Targeted tests green with `CHLU_GEN_DATA_ROOT` set; passed / skipped / failed, every skip named.
   ⚠ ⛔ **Do not start a full suite** — the Hub is running one.

## What would prove this spoke wrong
- ⭐ **Your random-code control scores well above chance.** Then the code-level probe is reading probe capacity, not
  store content, and every "the store drops it" conclusion is unsupported.
- You used a different probe at different points and called the difference a loss.
- You measured only inside the corner and reported it as the global loss profile.
