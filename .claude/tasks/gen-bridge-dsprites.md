# Task: gen-bridge-dsprites — does any of this survive a second dataset?

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-bridge-dsprites.md` + artefacts under `.claude/outputs/gen-bridge-dsprites/`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (all) · **`.claude/tasks/gen-clu-phase1-rulings.md` — R3 (the ceiling is a GATE, stop-and-return), ⭐ **R10 (every dSprites trap, and ⛔ why the CODE wins over the paper's prose)**, R11 (the realised counts), R15 (how the Shapes3D gate was measured), R16 (⛔ the gate reads the COMPOSITIONAL term, not the aggregate), R19.1 + R20 (the winning decoder recipe and the in-corner discrimination metric)** · `.claude/outputs/gen-feature-bridge.md` · `.claude/outputs/gen-decoder-lift.md`.
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Base:** `main @ be27713` · **Branch:** `agent/experiment-engineer/gen-bridge-dsprites` · **Worktree:** fresh (`../CHLU-gen-dsprites`).
- **Env:** main venv, ⛔ no `uv sync` in the worktree. ⭐ **Run pytest with `CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen`** — a worktree has no `.claude/`, and without it every real-data test **silently skips**. **Report passed / skipped / failed, never passed alone.**
- ⚠ **Budget ≈ 8 h — the longest in the wave, and it may need to run overnight.** dSprites train is **730,988** rows against Shapes3D's 479,928. ⭐ **Per-seed incremental writes and skip-if-exists are MANDATORY, not advisory.** ⛔ If budget runs out, ship **fewer seeds fully reported** and mark the rest **NOT RUN** — never a partial number presented as complete.

## ⭐ DIAL DECLARATION
- **Dial:** *the dataset.* ⛔ No arm, no CLU number, no kill-rule verdict — you are measuring whether the instrument stack exists on dSprites at all.
- **Laundering control:** the **matched-seen** corner, exactly as on Shapes3D. R16 made it a permanent companion row in every table in this phase; ⛔ it is not optional here.
- **Falsifies:** everything downstream on dSprites. ⭐ **If the decoder ceiling fails the gate, the correct outcome is STOP AND RETURN** — R3.1. ⛔ Do not run arms against a wall, and ⛔ do not tune toward a pass.

## The one job
Fit the encoder + decoder on **dSprites** and measure the **R3 decoder-ceiling gate** on the compositional term.
**One acceptance criterion: the gate verdict.** The fit is the means, not a second deliverable.

## Why this exists
⭐ Every number this programme has is from **one dataset**. *"Does it work on anything else?"* is the first question a
referee asks and we currently have no answer. ⭐ **dSprites is already fully wired** — the loader, the splits, the C set
and the A7 set all exist and are checksummed — and it has **never been run end to end.** This spoke is the cheapest
honest answer to that question, and it is a gate, so a negative is as useful as a positive.

## ⚠ dSprites is the SAME SHAPE of problem as Shapes3D — do not expect an easier metric
The pinned held-out filter includes **`latent_classes[:, shp] == 1`**, so the held-out cell pins **shape to ONE
value**, exactly as Montero's Shapes3D cell pins `shape == cylinder`. ⇒ ⛔ **The one-class problem carries over, and so
does its fix.** Use R20's metric **unchanged**:
> **Positives:** the held-out `shp == 1` items. **Negatives:** matched-seen **other shapes in the same corner**.
> **Score:** binary **balanced accuracy** (the gate statistic, R23.1), **AUROC as a mandatory companion, never the gate.**
⛔ Do not invent a new metric for this dataset. ⭐ Reusing the ruled one is what makes the two datasets comparable.

## ⛔ THE dSPRITES TRAPS — every one is documented and every one silently produces a wrong split
1. ⭐⭐ **The colour-column slice.** `load_sprites` **drops the leading colour column at load**:
   `latent_values[:, 1:]` and `latent_classes[:, 1:]`. The index constants `shp, scl, rot, tx, ty = 0,1,2,3,4` are
   valid **ONLY after that slice**. ⛔ Applying them to the raw 6-column arrays shifts every index silently and
   **yields an empty test set**. **Assert your realised test count and compare it to R11 before anything else.**
2. ⭐⭐ **The angle constants are SWAPPED relative to their names, and the filter only works because they are.**
   `a120 = 4π/3` (= 240°) and `a240 = 2π/3` (= 120°). So `rot < a120 AND rot > a240` is the band **[120°, 240°]**.
   ⛔ **It LOOKS like a bug. It is not.** An engineer "fixing" the apparent contradiction **inverts the held-out band**
   and every downstream number silently becomes a different experiment. ⛔ **Do not touch it.**
3. **Float boundaries.** The realised counts are boundary-sensitive: **13** rotation values are admitted, not the
   nominal 12; `> 0.6` admits **4** scale values. ⭐ **Realised count must be MEASURED and reported, never assumed.**
4. ⛔ **`latent_classes` vs `latent_values` is a literal collision** — the filter uses `latent_classes` for `shp` and
   `latent_values` for everything else. Mixing them typechecks and is wrong.
5. ⛔ **The paper's prose contradicts the code, and contradicts itself on rotation. THE CODE WINS** (R10). ⛔ Never
   re-derive the split from the paper.
6. ⛔ **Never repeat "the hold-out is 1 of 243 bins"** — it is ~1.9× that. The realised count is **6,292**.

## Reference values to check yourself against (⛔ measure, do not assume)
| | value |
|---|---|
| dSprites n_total / **n_test** / n_train | 737,280 / **6,292** / 730,988 |
| `c_invalid.json` sha256 | `10a39e713864c9f0c28f740caad49394cbda6f34f3d1d29bbc61dab9fc288666` |
| Shapes3D gate, for comparison | `ceiling_B / ceiling_A` on the compositional term, V1 decoder: **0.9414 ± 0.0179** balanced accuracy |
| the R3 floor | **0.8 ×** |

## ⭐ Start from the decoder recipe that WON — do not re-run the bake-off
R19.1 / R20: the winner was **`v1_capacity_schedule`** (capacity + a longer schedule, uniform pixel MSE untouched);
the cleverly-designed object-weighted loss **lost badly** (0.537). ⭐ **R20's standing lesson: default to capacity +
schedule first, before reaching for a designed objective.** ⛔ `dec_recipe` still defaults to `"baseline"`, the recipe
that **FAILED** the gate — **select the winner explicitly and assert the decoder parameter hash.** A baseline run
looks entirely normal in the logs and is **void**.

## ⛔ What you must NOT do
- ⛔ **Do not change the Shapes3D path.** Your changes are **additive**; **assert the Shapes3D bridge is bit-identical
  before and after** and report the assertion. Every number in this phase rests on it.
- ⛔ Do not regenerate `c_invalid.json`, `a7_redundant.json` or the splits. They are checksummed and single-sourced.
- ⛔ No arms, no store, no certificate, no kill-rule verdict. **Gate only.**
- ⛔ No cluster. CSF3 needs 2FA and is unreachable — emit a submit line and report it **NOT RUN**.

## Files you own
`chlu/experiments/exp_gen_bridge.py` · `chlu/core/gen_bridge.py` — ⭐ **additive only, dSprites path; the Shapes3D
path must not change behaviour** · `chlu/data/dsprites.py` · `tests/test_gen_bridge_dsprites.py` (new) ·
`chlu/config.py` — ⭐ **ONE new block, APPENDED AT EOF** (⛔ or extend the existing bridge block **only** if a new
field is genuinely required — say which, and why a new block would not do).
⛔ Nothing under `chlu/core/gen_store.py`, `gen_compose.py`, `gen_navigate.py`, `gen_certificate.py`,
`gen_request.py`, `gen_dream.py`, `chlu/eval/`, `scripts/csf3/`.
⚠ **`chlu/config.py` is shared with `gen-cell-harness` and `gen-navigation` this wave.** ⛔ **If you conflict, do NOT
merge hunk-by-hunk:** take the peer's whole file, re-apply your block, assert **0 deletions**.

## Acceptance
1. `PREREG.md` filed **before** the gate is measured, stating the predicted ratio and how it was derived.
2. Realised split counts **measured** and checked against 737,280 / 6,292 / 730,988; the C-set sha verified.
3. R3.3 **no-leak** assertion: held-out-combination images excluded from decoder training, **count reported, 0 leaked**,
   and the assertion **shown able to fire** on a deliberately seeded violation.
4. Decoder recipe `v1_capacity_schedule` selected explicitly; parameter hash asserted per seed.
5. The gate measured on the **compositional term** with the matched-seen normaliser and R20's in-corner metric;
   **balanced accuracy** as the gate, **AUROC** as companion; **PASS/FAIL stated plainly per seed and overall**.
   ⛔ **On FAIL: stop and return to the Hub. Do not proceed and do not tune.**
6. Shapes3D bit-identity assertion passes and is reported.
7. Per-seed incremental writes; any seed not completed reported **NOT RUN**, never as a zero.
8. Targeted tests green with `CHLU_GEN_DATA_ROOT` set, reported as passed / skipped / failed with every skip named.
   ⚠ ⛔ **Do not start a full suite.**

## What would prove this spoke wrong
- ⭐ **Your realised test count is not 6,292.** Then one of traps 1–4 has fired and every downstream number is a
  different experiment. **Check this before you fit anything** — it costs seconds and it is the whole ballgame.
- ⭐ **You "fixed" the angle constants.** The band inverts and the run is void.
- **The Shapes3D path moved.** Stop immediately; that voids the entire phase, not just this spoke.
