# c2w6-phi-leak-docstring — experiment-engineer report

**Task + acceptance criterion:** correct `StreamMemoryConfig.write_sign`'s false severance claim (it holds only at `atom_place_radius == 0`), propagate the qualifier to every other in-code instance, and pin the corrected behaviour with one regression test — comments + test only, no behaviour change.
**Status: done.** Suite green incl. K1's exact-zero probe and K2's fingerprint (both ran, neither skipped).
**Dial declaration:** none — docs-in-code + a regression test; this task measures nothing new. No prereg (§5 rule does not apply: no ratio/exponent/law is claimed; the 27 % figure is quoted from charter §A22, not re-measured).

---

## What I did

1. **Corrected the claim at 6 in-code sites** (all comments; `git diff` contains no executable change — see the ruff/test evidence below).
2. **Grepped the codebase for the same claim** (`sever`, `by construction`, `zero derivative`, `only channel`, `trains phi`, `settled_point`, `d q*/d q0`) — **full hit list with disposition in §Findings.**
3. **Added one regression test** to `tests/test_anti_erosion.py`, reusing the K1 rig rather than a new harness.

## How I verified

**(a) The mechanism, before writing anything** (`.claude/scratch/c2w6-phi-leak-docstring/probe.py`, `PYTHONPATH=… uv run --no-sync python`). Write-channel gradient into φ's output isolated by holding the read query at a fixed constant (`dL/dz` with `read(st, q_fixed)`, so the query path is closed and only the store state carries `z`):

```
r0.3 P1 off      |dL/dz| write-channel = 9.359511e-04  all_exact_zero=False
r0.3 P1 on       |dL/dz| write-channel = 0.000000e+00  all_exact_zero=True
r0.0 P1 off      |dL/dz| write-channel = 0.000000e+00  all_exact_zero=True
r0.0 plainSGD    |dL/dz| write-channel = 6.834356e-09  all_exact_zero=False
r0.3 plainSGD    |dL/dz| write-channel = 4.029910e-06  all_exact_zero=False
```

This reproduces the charter §A22 finding in miniature and is exactly the docstring's claim structure: the sign write **does** sever the write channel bitwise at radius 0 (row 3), the plain-SGD diagnostic **does** open it (row 4, 6.8e-09 — the "channel (ii)" `gradient_probe` documents), and **placement re-opens it by ~5 orders of magnitude regardless of the sign write** (row 1 vs row 3).

**(b) Tests.** `uv run --no-sync pytest …`
- `tests/test_anti_erosion.py` → **19 passed** in 99.11s (0 skipped) — includes the new test.
- `tests/test_anti_erosion.py tests/test_blocks.py` (re-run after the final edits) → **41 passed** in 165.15s.
- `tests/test_blocks.py tests/test_placement_probe.py` → **44 passed** in 96.71s.
- Ship-condition check, explicit: `-k "k1 or k2 or placement"` → **8 passed, 0 skipped**, collection confirms `test_k2_reference_fingerprints_from_before_the_build` and `test_k1_the_partition_zeroes_the_depth_determining_leaves_bitwise` are among them. **K1 and K2 are GREEN; no fingerprint was touched.**

**(c) Lint.** `uv run --no-sync ruff check chlu/core/blocks.py chlu/training/train_cluformer.py tests/test_anti_erosion.py` → **All checks passed!**
⚠ `ruff format --check` reports all three files "would reformat" — **pre-existing**: I verified by piping `git show HEAD:<file>` through `ruff format --check` and all three were already non-conformant at the base commit. The repo gates on `ruff check`, not `ruff format`; I did **not** reformat (out of scope, and it would be a large foreign diff).

## Findings

### The correction (what the comments now say)
The severance is stated as **conditional on `atom_place_radius == 0`**, with H1b's localized placement named as the live exception in the shipped run-1/2/3 config (`atom_place_radius = 0.3`), the 27 % measurement (0.0908 → 0.0659, charter §A22) cited, and `erosion_partition = True` named as the closure. Kept to the repo's comment idiom (a constraint the code cannot show), not a change log.

### Grep hit list — every hit and its disposition
Searched `chlu/` for `sever`, `by construction`, `zero derivative`, `only channel`, `trains ``phi```, `settled_point`, `d q*/d q0`, `jnp.sign`. Complete list of φ/write/sign-relevant hits:

| Site | Text | Disposition |
|---|---|---|
| `core/blocks.py:490-496` `write_sign` | "sign-SGD severs `d(store state)/d(phi)` … *only* channel to `phi` **by construction as well as by theorem**" | ✅ **CORRECTED** — the primary false claim. Now: severs the *inner loop* only; ⛔ block naming placement as the live exception + 27 % + `erosion_partition`; cites the new test by name. |
| `core/blocks.py:422-426` Tier-III module docstring, **design rule 1** | "⇒ the block trains end-to-end **only** through the trajectory read. `read_mode="settled_point"` is retained solely as the pre-registered *control* that must measure ~0" | ✅ **CORRECTED** — this is the program-level statement of the same claim and arguably the more load-bearing one (it is the design rule everything downstream cites). ⚠ clause added: true of the READ; whole story only at radius 0; settled-point control has a non-zero floor at 0.3. |
| `core/blocks.py:467-468` `read_mode` field | "`"trajectory"` is the only mode that trains `phi`… `"settled_point"` exists as the pre-registered **zero-gradient control**" | ✅ **CORRECTED** — "only READ mode"; zero-gradient only at radius 0. |
| `core/blocks.py:541-555` `atom_place_radius` | (no claim, but the lever itself) | ✅ **AUGMENTED** — the lever now names its own gradient consequence, since this is where a reader of the placement knob looks. |
| `core/blocks.py` `_write_stages` step 1b inline comment | "adds no parameter and no state byte; …C3 locality holds" (true, but silent on the gradient) | ✅ **AUGMENTED** at the code site: the assignment is differentiable in `z` and sits outside the sign-gated loop. |
| `core/blocks.py` `read()`, `mode == "settled_point"` branch | "the pre-registered **ZERO-GRADIENT control**" | ✅ **CORRECTED** — zero on the READ path only; end-to-end zero-gradient only under `erosion_partition`. |
| `training/train_cluformer.py:1186-1197` `gradient_probe` docstring | "the trajectory read is **the only channel** that does" | ✅ **CORRECTED** — "only channel *through the read*"; ⚠ clause: at radius > 0 the settled-point arm's `grad_phi` is a non-zero floor, so the reported `ratio_traj_over_point` **UNDERSTATES** the read's share unless the partition is on. *(Note for the analyst: every banked S2 ratio measured at the run-1/2/3 config inherits this — it is a conservative bias, not an inflation.)* |
| `training/train_cluformer.py:1227-1230` two-causes comment | "(ii) sign-SGD's zero derivative, which **severs** `d(store state)/d(phi)`" | ✅ **CORRECTED** — severs the *inner loop*'s; ⚠ "exactly zero" presumes radius 0; placement is a third channel neither (i) nor (ii) closes. |
| `core/blocks.py:620-622` `erosion_partition` | "only the write channel is severed" | ⬜ **JUDGED FINE, no edit.** True as written, and it is the *closure* side of the claim. |
| `experiments/exp_anti_erosion.py:112` | "the one channel P1 severs is also the one that gave the wells a reason to exist" | ⬜ **JUDGED FINE.** Refers to the store-content gradient channel, which P1 genuinely severs. |
| `core/implicit_grad.py:33-40` | "a settled-point read-**out** sends no gradient to `phi`" | ⬜ **JUDGED FINE, deliberately not edited.** The sentence is scoped to the read-out operator, which is exactly the true scope; `implicit_grad.py` knows nothing of the streaming write and importing the caveat there would be wrong-layer. Listed for the curator because a reader *could* over-read it as the block-level claim. |
| `experiments/exp_trajectory_read.py:798` "severs the path to `q0`" | truncation-direction warning in the probe rig | ⬜ **JUDGED FINE.** Different mechanism (tail truncation + `stop_gradient`), different rig, φ trained through the read only. |
| `core/factored_store.py:697` "because `phi` is frozen, `dq*/dq0 = 0` is discharged **by construction**" (+ `:693`, `:780` reference scales) | cat-test rig | ⬜ **JUDGED FINE.** φ is genuinely frozen there, so no write channel exists. |
| `core/blocks.py:588`, `:736`, `:1815`, and the ~45 other `by construction` hits in `chlu/` | byte ledgers, C3 locality, partitions of unity, equal-compute, CL-baseline collapse, … | ⬜ **JUDGED FINE** — unrelated to the φ/write/sign claim; not touched (scope discipline). |

**Curator hand-off (§A23.3's other half):** the qualifier that must propagate program-wide is *"the trajectory read is the only channel to φ" is a statement about the READ; at `atom_place_radius > 0` the write is a second channel (27 % of layer-0's φ gradient) that only `erosion_partition` closes.* The three highest-value non-code sites to check are anything quoting **design rule 1**, anything quoting the **`0.0 / 2.654e-9 / 6.421e-3`** reference triple as an end-to-end statement, and any text calling `settled_point` the **"zero-gradient control"**.

### The new test
`tests/test_anti_erosion.py::test_the_placement_path_is_a_live_gradient_channel_to_phi` (+ helper `_write_channel_grad_into_phi`). Three arms at the module's existing run-2 toy rig (`_mcfg`, `atom_place_radius=0.3`, `write_margin=0.6`, seed key 2, z seed 7, fixed query seed 99):
- (a) radius 0.3, partition **OFF** → `‖dL/dz‖ > 0` (the leak is live) — **9.36e-04** as measured above;
- (b) same config, partition **ON** → **exactly 0.0 bitwise** (`np.all(g == 0.0)`);
- (c) radius **0.0** → **exactly 0.0 bitwise** — the condition the corrected docstring now states.

Isolation: the read is launched from a *fixed constant* query, not from `z`, so this does **not** duplicate `test_k1_the_query_gradient_into_phi_survives_the_partition` (which deliberately keeps both paths open) — every route from `z` to the loss runs through the store state the write produced.

## Git footprint

- **Branch:** `agent/experiment-engineer/c2w6-phi-leak-docstring`, off `main @ d1149a4`. **No worktree** (main checkout was clean: `git status` empty, `main` checked out, no foreign uncommitted work). Rebased onto local `main` — no-op, base has not moved. **Not pushed** (neither `origin` nor `clu-dev`), per task.
- **Commits:**
  - `2f67b3c` `[experiment-engineer] qualify the phi-severance claim: it is conditional on atom_place_radius == 0` — `chlu/core/blocks.py`, `chlu/training/train_cluformer.py`
  - `2d3a843` `[experiment-engineer] pin the placement leak so the docstring cannot drift back` — `tests/test_anti_erosion.py`
- **Files touched (3):** `chlu/core/blocks.py` (+40/−5, comments only), `chlu/training/train_cluformer.py` (+17/−5, comments only), `tests/test_anti_erosion.py` (+45/−0).
- **Behaviour:** none. The two source diffs are entirely `#:`/`#`/docstring lines; K2's bit-identity fingerprints from `main @ 104ca19` still reproduce (that test passed), which is the strongest available evidence that no bit moved.
- **Unresolved conflicts:** none.

## Open questions / follow-ups / risks

1. **`gradient_probe`'s banked S2 numbers.** The corrected docstring implies every `ratio_traj_over_point` measured at a run-1/2/3 config (radius 0.3, partition off) has a **non-zero denominator floor from the write**, i.e. the published ratio is a *lower bound* on the read's share. Nobody owns re-checking whether any reported S2 ratio was described as "the settled-point arm reads 0". Cheap fix if wanted: re-run `gradient_probe` once with `erosion_partition=True` and report both.
2. **`ruff format` is not clean repo-wide** (pre-existing on all three files I touched). If the Hub wants `ruff format` as a gate, that is a separate, large, mechanical task — deliberately not done here.
3. I did **not** touch `multiplicity_read.py`, `monitors.py`, `experiment_cmd.py`, `psi_readout.py` (excluded), nor `implicit_grad.py` (judged correct at its own layer — see table).

## Proposed handover updates (for the Hub)

- **New §7 entry (mechanism, standing) — 7.30 `atom_place_radius > 0` is a WRITE-side gradient channel to φ.** H1b's localized placement assigns φ's output into the slot's atom centers outside the sign-gated inner loop, so sign-SGD's zero derivative does **not** make the trajectory read the only channel to φ at the shipped run-1/2/3 config (`atom_place_radius = 0.3`). Measured: φ's layer-0 gradient 0.0908 → 0.0659 under P1 = **27 % flowing through the write** (§A22); reproduced at cell scale as 9.36e-04 → exactly 0.0 (`c2w6-phi-leak-docstring` probe). `erosion_partition = True` is the only closure. **Consequence:** `settled_point` is a zero-gradient control only at radius 0, and `gradient_probe`'s traj/point ratio understates the read's share at radius > 0. In-code claims corrected at 6 sites on `agent/experiment-engineer/c2w6-phi-leak-docstring`; **prose/paper sites are the curator's half of §A23.3** (check: design rule 1, the `0.0 / 2.654e-9 / 6.421e-3` triple, "zero-gradient control").
- **No config default changed.** `write_sign` stays `True`, `atom_place_radius` stays `0.0`, `erosion_partition` stays `False`.
- **§7.23 unaffected** — I added a function-scoped-safe test to a file whose x64 fixture is `scope="module"`; the new test does not touch x64 and the module still passes standalone and alongside `test_blocks.py`/`test_placement_probe.py`.
