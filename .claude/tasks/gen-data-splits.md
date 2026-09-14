# Task: gen-data-splits — Shapes3D + dSprites loaders, the factor grid, and the Montero splits with a hold-out assertion that can fail

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-data-splits.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (all of it) · **`.claude/outputs/gen-montero-splits-spec/montero_splits.json`** (your specification — it is binding) · `chlu/data/mnist.py` and `chlu/data/industrial/base.py` for the loader conventions you must match · `chlu/cli/data_cmd.py`. ⛔ Read no campaign ledger, charter or handover.
- **Base:** `main @ b0257bc` · **Branch:** `agent/experiment-engineer/gen-data-splits` · **Worktree:** yes, fresh (`../CHLU-gen-data`). ⛔ Do not reuse or touch `CHLU-bprime-family2`, `CHLU-wt2`, `CHLU-wt3`, `CHLU-merge` — parked, not free.
- ⛔ **PRECONDITION — mechanical, not a judgement call.** Do not start until
  **`.claude/outputs/gen-montero-splits-spec/montero_splits.json` exists** and its `shapes3d.condition.status`
  reads `PINNED` or `RECONSTRUCTED`. If the file is absent, **stop and report** — do not reconstruct the split yourself.
  If it reads `NOT OBTAINED`, **stop and report** — that is a Head decision, not yours.

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** *none — instrument.* ⛔ No CLU number, no store, no encoder, no generation, no performance claim.
- **Laundering control:** N/A (nothing is measured).
- **Falsifies:** N/A — but see the positive control below. **An assertion that cannot fail proves nothing.**

## The one job
Ship Shapes3D and dSprites loaders with the exhaustive factor-grid index and the Montero
recombination-to-element splits, such that **every held-out combination is verified absent from train by an
assertion that is itself proven capable of firing.**

## Why the positive control is the real deliverable
The entire B cell rests on "this combination was never seen in training." If that assertion is vacuous — wrong axis
order, wrong comparison, an empty held-out set — it will pass silently and every downstream number is void, and we will
not find out until a referee does. ⭐ **So the acceptance criterion is not "the assertion passes." It is "the assertion
passes on the real split AND fires on a deliberately corrupted one."**

## Files you own (nobody else touches these)
- `chlu/data/shapes3d.py` · `chlu/data/dsprites.py` · `chlu/data/gen_splits.py` — all new.
- `chlu/cli/data_cmd.py` — **append two subcommands only** (`chlu data shapes3d`, `chlu data dsprites`), matching the
  existing `cmd_data_mnist` shape. ⛔ Touch no existing line.
- `chlu/config.py` — **shared file, minimal hunk.** Append **one** dataclass (`GenDataConfig`) at the end and **one**
  field on `CHLUConfig`. ⛔ Nothing else. Report the exact line ranges you touched.
- `tests/test_gen_data_splits.py` — new.

## Data rules (binding)
- Datasets live at **`.claude/data/gen/shapes3d/`** and **`.claude/data/gen/dsprites/`**. ⛔ Never committed, never
  backed up, no large binary enters git.
- ⭐ **Record source URL, licence, byte size and checksum for each file. That record IS the backup** — it is what makes
  the download reproducible, so it goes in your report as a table, not just in a log.
- The exhaustive factor-grid index is a derived artifact: build it, store it under `.claude/data/gen/<dataset>/`, and
  record its checksum too.
- Also record the **cluster staging path and the command that puts each dataset there**; you need not execute the
  staging, but the recipe must be in your report.

## What to build
1. **Loaders** matching house conventions: config-driven, seeded, `--quick` honoured, deterministic, memmap-friendly.
   Shapes3D is a single large HDF5 — do not load it whole into memory as a matter of course.
2. **Factor grid index:** for each dataset, the ordered factor names, cardinalities, value grids, and a bijection
   between a factor-value tuple and its dataset row. ⚠ **Verify the axis ordering empirically against the raw file
   rather than trusting the documented order** — decode a handful of known rows and check the factors are where you
   think. Report the check.
3. **Splits** exactly as `montero_splits.json` specifies, per dataset. ⛔ If the spec and the data disagree
   (counts, cardinalities, ordering), **stop and report the discrepancy. Do not adjust the spec to fit the data
   or the data to fit the spec.**
4. **The hold-out assertion**, run at load time, over the full train split, reporting the count of combinations checked.
5. ⭐ **The positive control.** Inject a known held-out combination into the train index and show the assertion **fires**.
   Do this as a test, not a one-off. An assertion that has never been seen to fail is not evidence.

## The C set — build it here, in factor space, once, for every arm
Later spokes need a set of **structurally invalid** factor requests. Define and emit it now, from the factor grid alone:
(a) two different values of the same factor requested simultaneously; (b) a factor value outside that factor's range.
⛔ **It is defined in factor space and is identical for every arm.** It must not reference wells, modes, clusters or
any model-internal structure — those do not exist yet and two of the four arms will never have them.
Emit it as `.claude/data/gen/<dataset>/c_invalid.json` with a documented schema and a checksum.

## Acceptance
1. `chlu data shapes3d` and `chlu data dsprites` fetch, verify checksums, and build the index.
2. Splits reproduce the spec's `expected_counts`, or the disagreement is reported and nothing is silently adjusted.
3. The hold-out assertion passes on the real split, with the count of combinations checked reported.
4. ⭐ The positive control fires. This is the criterion the task lives or dies on.
5. `c_invalid.json` emitted for both datasets, in factor space, with schema and checksum.
6. Provenance table: source, licence, bytes, checksum for every file, plus the cluster staging recipe.
7. Suite green against the named base. Record the pass count before and after using the **main venv** per protocol §4
   (`PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python -m pytest -q`). ⚠ A fresh worktree venv drops ~7
   tests via optional extras — not a regression, but say so if you see it rather than reporting a smaller number.
8. Re-verify `git log --oneline main..agent/experiment-engineer/gen-data-splits` **from the main repo** before removing
   the worktree.

## What would prove this spoke wrong
- **The assertion passes and the positive control does not fire.** Then the assertion is vacuous and every B number
  built on it is void. Check this before you believe anything else in your own report.
- The factor axis order is off by one and every "held-out" combination is actually a different combination. Item 2's
  empirical decode check is what catches it — do it properly.

---

# ⛔ TRAPS — folded in from `gen-montero-splits-spec` on 2026-09-09, BEFORE this spoke spawned
**Also read `.claude/outputs/gen-montero-splits-spec.md` in full.** These were found by measuring, not by reading, and
every one of them fails *silently*.

## T0 — The paper's prose is NOT the spec. The code is.
The splits are pinned from the official implementation `mmrl/disent-and-gen@bff35a7570d08fdef0d10c21b093b9a24d98dde6`,
file `scripts/configs/datasplits.py`. **The paper's prose contradicts the code for both datasets and contradicts itself
on dSprites rotation.** ⛔ Implement `montero_splits.json`. Do not "check it against the paper" and adjust — the
disagreement is already adjudicated and the code won.

## T1 — dSprites colour-column slice (severity: produces a SILENTLY EMPTY test set)
`load_sprites` slices the colour column off at load: `latent_values = data_zip['latents_values'][:, 1:]`.
So `shp,scl,rot,tx,ty = 0,1,2,3,4` are valid **only post-slice**. Applied to the raw 6-column npz the filter selects
**nothing** — colour class is always 0, so `== 1` is empty. ⭐ **Assert the test set is non-empty and of the expected
size before anything else runs.**

## T2 — dSprites angle constants are bound to SWAPPED values (severity: an engineer "fixing" it inverts the band)
```
a90, a120, a180, a240 = np.pi / 2, 4 * np.pi / 3, np.pi, 2 * np.pi / 3
```
`a120` = 4π/3 = **240°** and `a240` = 2π/3 = **120°**. So `(rot < a120) & (rot > a240)` is **not** impossible — it means
**120° < θ < 240°**, which reconciles with the paper. ⛔ **Do not repair this apparent bug.** Reproduce the constants
verbatim, with this comment next to them. Verified by two independent fetches via different CDNs.

## T3 — dSprites float-boundary hazard (severity HIGH; the count is decided by round-off, not intent)
Two thresholds land exactly on grid values: `scale > 0.6` (⇒ 4 or 5 scale values) and 120°/240° landing exactly on
`np.linspace(0, 2π, 40)` indices 13 and 26 (⇒ 12, 13 or 14 rotation values). `n_test` is therefore
**5,808 / 6,292 / 6,776** depending on float64 round-off.
⭐ **Mandatory: do NOT regenerate the grid and reason about nominal values. Evaluate the filter verbatim against the
actual npz array (post `[:, 1:]` slice) and report the REALISED count, reconciled against the scout's 5,808.**

## T4 — Shapes3D: the whole held-out set is 72 images
Test filter: `(oh>=0.8) & (wh>=0.8) & (fh>=0.8) & (scl>=1.1) & (shp==1) & (rot>20)` ⇒
`n_test = 2·2·2·3·1·3 = 72` of 480,000; `n_train = 479,928`. **Hub-recomputed independently — it is 72.**
Assert exactly this count. Hazard here is LOW (no boundary ties), but ⛔ **if you get any number other than 72, stop and
report — do not adjust the filter to reach it.**

## T5 — The two datasets do NOT share a rule, and one literal means two things
recomb-to-**element** uses `latent_classes[:,shp]==1` → **ellipse**; recomb-to-**range** uses
`latent_values[:,shp]==1` → **square**. Different arrays, same literal, different shapes. Carry nothing across.

## Amended acceptance (supersedes item 2 above)
- Realised counts reported per dataset: Shapes3D **must** be 72; dSprites is **whichever of 5,808 / 6,292 / 6,776 the
  data actually yields**, reported as realised with the float-boundary reconciliation shown. ⛔ Neither is adjusted to
  match an expectation.
- The non-empty assertion (T1) runs before any other check.
- The angle constants (T2) appear verbatim in the code with the swap documented in a comment.
