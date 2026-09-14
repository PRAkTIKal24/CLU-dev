# Task: gen-navigation — is there structure BETWEEN the codes, now that we know the commitment lands on nine?

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-navigation.md` + artefacts under `.claude/outputs/gen-navigation/`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (all) · **`.claude/tasks/gen-clu-phase1-rulings.md` — R20 (the 30-bit design constant), R23.5 (the store ceiling), R26.3 (⛔ why clamp-and-relax was RETIRED — do not re-invent it), R29 pivot items 1 and 4, R31 (the corrected collapse mechanism)** · `.claude/outputs/gen-commitment-collapse.md` · `.claude/outputs/gen-external-classifier.md`.
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Base:** `main @ be27713` · **Branch:** `agent/experiment-engineer/gen-navigation` · **Worktree:** fresh (`../CHLU-gen-nav`).
- **Env:** main venv, ⛔ no `uv sync` in the worktree. ⭐ **Run pytest with `CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen`** — a worktree has no `.claude/`, and without it every real-data test **silently skips** and your green line means less than it looks. **Report passed / skipped / failed, never passed alone.**
- ⚠ **Budget ≈ 5 h.** ⭐ Write per unit, skip any unit whose artefact exists. ⛔ Never hold to the end.

## ⭐ DIAL DECLARATION
- **Dial:** *navigation* — whether the store's landscape supports meaningful movement **between** wells/modes.
- **Laundering control:** ⭐ **straight-line interpolation between the two endpoint READ vectors, store bypassed.** On every number. See below — it is the control most likely to kill this.
- **Falsifies:** the hypothesis that the store carries traversable structure. ⛔ **A clean negative is a result** and routes to the Head, not to re-tuning.

## Why this exists, stated honestly
`gen-commitment-collapse` **FAILED** its pre-registered gate: the `argmax` commitment lands 72 held-out requests on
**9** codes, and no stopping rule, multiplicity or temperature setting fixes it. ⭐ **R29 pre-registered that a fail
routes to new approaches — "particle traversal etc." — not to abandonment.** This is that spoke, and it is also
pivot item 4 in the Head's own order.
⭐ **The question the collapse leaves open:** the commitment lands on nine *points*. **Is the space between them
structured?** If the landscape supports smooth, semantically ordered traversal, that is a mechanism result which
**survives** the commitment failure. If it does not, we have bounded the store's usefulness and we say so.

## The one job
Traverse from one store mode to another, decode at each step, read every intermediate image with the **gated external
classifier**, and say whether the factor readings change **smoothly and monotonically** along the path.

## ⛔ What you must NOT re-invent
R26.3 **retired clamp-and-relax**: the chain samples a mode, the clamp does **not** steer, and the coupling-off control
took all of the apparent gain (`steering-attributable = +0.0015 ± 0.0204 = ZERO`). ⛔ **Do not rebuild conditional
dreaming as steering.** ⭐ This spoke is about the **geometry between modes**, which that work never measured.

## ⭐⭐ THE CONTROL THAT IS MOST LIKELY TO KILL THIS — build it FIRST
For every traversal, run the identical evaluation on a **straight line in read space**: take the two endpoint read
vectors, linearly interpolate, decode, classify. **The store is not consulted at any intermediate point.**
- ⭐ **If linear interpolation traverses as well or better, the store's landscape adds nothing** and "navigation" is
  just interpolation with machinery attached. ⛔ **Report that plainly; it is the honest outcome and it is publishable.**
- **Second control:** the same traversal on a **deleted store** (`delete_store`, `mean` mode — the fair deletion). Any
  smoothness that survives deletion is smoothness the store never supplied.
- ⛔ Both controls run on **every** number, not as a footnote.

## ⛔ THE GUARANTEE — the easiest way to void the programme's biggest result
`read_from_codes(store, codes)` takes **no features**; the only encoder → decoder channel is **30 bits** (R20).
⛔ **`soft_read(store, X, tau)` takes FEATURES — using it anywhere on the traversal path voids the no-bypass
guarantee.** Every intermediate point is derived from **codes / store state**, never from a feature tensor.
**Assert the path touches no feature tensor**, exactly as `gen-read-compose` and `gen-commitment-collapse` did.
⭐ Report the **realised channel width** at every intermediate point; ⛔ any widening beyond 30 bits is a **declared
change to the claim**, not a free improvement.

## The legs
1. **Controls first** (linear interpolation + deleted store), with the evaluation harness they share.
2. **Endpoint selection, pre-registered.** ⛔ Do not hand-pick flattering pairs. File the rule in `PREREG.md`:
   which mode pairs, how many, chosen by a stated criterion (e.g. all pairs differing in exactly one factor), and
   ⭐ **include pairs differing in `scale` and `orientation`** — R31 established those are the factors the code
   **drops** inside the corner, so they are where traversal is most likely to fail. ⛔ Testing only the easy axis
   (hues) would be scoring your own exam.
3. **Traverse and measure.** At each step: decode → external classifier → record every factor reading.
   Report **monotonicity** of the target factor, **stability** of the non-target factors (a traversal that changes
   everything is not navigation), and **step-to-step image distance** (does it move smoothly or jump?).
4. **The honest summary statistic**, pre-registered before you look: what fraction of pre-registered pairs traverse
   monotonically in the target factor **and** beat both controls.

## ⛔ Live traps
1. **`dec_recipe` defaults to `"baseline"`** — the decoder that FAILED the gate. Winner **`v1_capacity_schedule`**.
   ⭐ Select explicitly and **assert the decoder parameter hash**. A baseline run looks normal in the logs and is **void**.
2. ⭐ **Use the GATED external classifier** (`classifier_gate.json` = `PASS`), identical at every step and for every
   control. ⛔ Never the linear probe — it shares the decoder's lineage.
3. ⚠ **Feature caches are non-uniform:** seed 0 is `scope="full"`; seeds 1–2 are `scope="used"` and contain **none**
   of the matched-seen rows.
4. ⚠ **A per-factor statistic is meaningless without `n_values_realised`.** Montero's corner pins `shape` to ONE value,
   and a factor with one realised value yields a confident-looking zero. ⭐ **Report `n_values_realised` beside every
   per-factor number.** This cost the last spoke a wrong headline.
5. **JAX cold start ~20 min. 16 GB machine** — ⛔ never start a full suite; ⛔ never hold two feature caches.

## Files you own — all NEW
`chlu/core/gen_navigate.py` (new) · `chlu/experiments/exp_gen_navigate.py` (new) ·
`tests/test_gen_navigate.py` (new) · `chlu/config.py` — ⭐ **ONE new block, APPENDED AT END OF FILE.**
⛔ Nothing under `chlu/core/gen_store.py`, `gen_compose.py`, `gen_bridge.py`, `gen_certificate.py`, `gen_request.py`,
`gen_dream.py`, `chlu/eval/`, `chlu/data/`, `scripts/csf3/`.
⚠ **`chlu/config.py` is shared with `gen-cell-harness` this wave** — it also appends at EOF. ⛔ **If you conflict,
do NOT merge hunk-by-hunk:** take the peer's whole file, re-apply your block, assert **0 deletions**. That file's
syntax has been broken twice this way.

## Acceptance
1. `PREREG.md` filed **before** any traversal is measured: the endpoint-selection rule, the monotonicity definition,
   and the summary statistic.
2. Both controls implemented and run on **every** number.
3. No-feature-tensor assertion passes; realised channel width reported at every intermediate point.
4. Decoder hash asserted; the gated external classifier used identically across arms and controls.
5. Traversal results by factor, **with `n_values_realised` beside each**, and the pre-registered summary statistic.
6. Per-unit incremental writes; anything not completed reported **NOT RUN**, never as a zero.
7. Targeted tests green, run **with `CHLU_GEN_DATA_ROOT` set**, reported as passed / skipped / failed with every skip
   named. ⚠ ⛔ **Do not start a full suite.**

## What would prove this spoke wrong
- ⭐ **Linear interpolation between the endpoint reads traverses as well.** Then the store's landscape is not doing the
  work. This is the single most likely outcome and it must be reported as the headline if it happens.
- ⭐ **Traversal is smooth only on the hue axes** — the factors the code already tracks — and fails on `scale` and
  `orientation`. That would mean navigation inherits the commitment collapse rather than escaping it.
- You hand-picked the mode pairs. The pre-registered selection rule exists to stop exactly that.
