# w10 spoke/seeds-and-tables — LOG

START_UTC: 2026-09-21T08:26:07Z
Branch: w10/seeds-and-tables · worktree C:/chlu/worktrees/CHLU-w10-seeds-and-tables
Scored runs: NONE (by charter). This spoke re-reads artefacts only.

## 08:26Z — orientation
- Read `/c/chlu/tasks/w10-spoke-seeds-and-tables.md`.
- Read `/c/chlu/evidence/outputs/advisor-wave-plan-w8-w10.md` and
  `/c/chlu/evidence/outputs/w8-advisor-report.md`.
- Evidence dirs located: `/c/chlu/outputs/{w8-downdate,w8-encoder-loop,w8-env-scout,
  w8-novelty,w8-single-design,w9-harness,w9-maze}` plus `/c/chlu/evidence/outputs/`.

## 08:40Z — artefact survey complete, sources fixed
Machine-readable sources found (all read, none written):
- `/c/chlu/outputs/w9-maze/runs/scored{1,2,3}_seed{0,1,2}_test.json` — CMAZE-1 hub, TEST, 3/3 scored
  seeds. Carries `table`, `env_columns`, `display`, `ledger`, `representation_ceiling`,
  `pipeline_ceiling_rows`, `span_ceiling_rows`, `baseline_params`, `lookup_stored`.
- `/c/chlu/outputs/w9-maze/runs/amend_exactdelete_seed0_val.json` — Amendment 1, VAL, seed 0.
- `/c/chlu/outputs/w9-maze/replication/replica_seed0_test.json` — independent replication of S1.
- `/c/chlu/outputs/w8-encoder-loop/CROSS-ARM.json` — Shapes3D rows 3/4, 3 seeds, arms
  alone/loop_nosw/loop_swap. `/c/chlu/outputs/w8-encoder-loop/run/encoder_loop_report.json` — pins.
- `/c/chlu/outputs/w8-downdate/DOWNDATE-READY-test.json` — exact-deletion assertions, 3 seeds, TEST.
- `/c/chlu/outputs/w9-harness/harness_seed{0,1,2}.json` — CMAZE-1 60k-param tier, 3 seeds.
- `/c/chlu/outputs/hub-generation-table/relaunch/scored/scored_spec.json` + `LOG-iter2.md` —
  Shapes3D incumbent cell, **seed 0 only**.

### Pin split decided (charter: different budgets do not share a table)
| table | instrument | pins |
|---|---|---|
| T1 | CMAZE-1, hub/maze | K=8 A=31 ridge=0.01, AE 12000, 410,688 read floats, test |
| T2 | Shapes3D, w8/encoder-loop | `w8_v1_short` decoder, joint_steps=400, n_train=8192, dec 400 |
| T3 | CMAZE-1, w9/harness tier | 60k-param arms, 500 train / 200 eval, enc 1200 / bc 2000 |
| T4 | exact deletion | two implementations, w8/downdate TEST + w9/maze TEST |
| NR | Shapes3D incumbent cell | K=64 A=1216, eval_b — **seed 0 only, NOT 3 seeds** |

## 09:05Z — builder + figures
- Wrote `build_tables.py`. It reads ONLY the paths in its `SOURCES` register, emits `TABLES.json`
  (every cell carries `provenance` = [{source, path, pointer}] one entry per seed) and `figures/`.
- Loaded the `dataviz` skill before writing any chart code. Palette = the skill's validated default;
  ran `validate_palette.js` on the two subsets used:
  - 3-slot all-pairs `#2a78d6,#eb6834,#1baf7a` → ALL CHECKS PASS (worst all-pairs CVD ΔE 9.2,
    normal-vision ΔE 24.0); one contrast WARN on aqua ⇒ relief rule applied (direct labels + the
    table view in TABLES.md).
  - 8-slot adjacent → ALL CHECKS PASS (worst adjacent CVD ΔE 9.1, normal-vision ΔE 19.6).
- Rendered and eyeballed each figure; fixed three real layout defects (fig1 crushed rows 2–4 under
  row 1 ⇒ faceted onto two panels rather than a dual axis; fig3/fig4/fig5 legends and reference-line
  labels collided with marks ⇒ moved outside the plot area / given surface-coloured relief).

## 09:40Z — TABLES.md rendered from the same manifest
- `render_md()` builds `TABLES.md` from the object that is serialised to `TABLES.json`, so the prose
  and the JSON cannot drift apart.
- T1 split into 1a RESULT rows / column-class legend / 1b ceilings / 1c caveat columns /
  1d DISPLAY rows (three sub-tables) / 1e instrument audits.

## 09:55Z — acceptance verification
- Wrote `verify_tables.py`. It does NOT import the builder: it re-reads `TABLES.json`, re-opens each
  source at the recorded path, re-resolves each `provenance` pointer and asserts the manifest value
  equals the value on disk; it also re-hashes every source and every figure.
- ⛔ FAILED FIRST on two real defects of mine, both fixed:
  1. `/c/chlu/outputs/w9-maze/replication/REPLICATION.md` was cited in TABLES.md but not registered
     as a source. Registered.
  2. I printed `scored{1,2,3}_seed{0,1,2}_test.json`, which expands to 9 files when only 3 exist
     (`scored1_seed0`, `scored2_seed1`, `scored3_seed2`). Rewritten as
     `scored{1_seed0,2_seed1,3_seed2}_test.json`.
- ⭐ PASS: 27 sources hashed, **240 cells**, **720 per-seed values re-resolved from disk**, 5 figures
  hashed, 5 declared NOT-RUNs each with reason + launch line, 4 corrections.
- ⭐ Determinism: `sha256sum figures/*.png` before and after a second `python build_tables.py` —
  zero diff. Builder wall time ~1 s, foreground, no background job at any point.

## 10:05Z — corrections found while tracing (all recorded in TABLES.md §CORRECTIONS)
- **C1** the task file's `0.9037–0.9130` band mixes a 3-seed mean with a per-seed value, and the
  artefact says **0.9128**, not 0.9130 (`scored3_seed2_test.json`). Per-seed minimum is 0.8905.
- **C2** ⛔ the w8 advisor report's exact-deletion range `1e-16…1e-14` understates the worst observed
  deviation by ~5 orders of magnitude: `worst_max_abs_dev_over_seeds = 1.1255e-09` (TEST). The claim
  survives on the artefact's own bar (`worst_in_float32_ulp_at_unit_scale = 0.009442`,
  `passed_every_seed = true`) but the quoted range is wrong.
- **C3** the n-way row's denominator (183/192/191, conjunction) differs from its ceiling's
  (389/403/414, per route). Both readings printed.
- **C4** `Wave 9 inherits the 0.4980 blind-sequential column` mis-attributes it: at the harness tier
  BLIND-SEQUENTIAL is **0.5317** and FRAG r1/r2 are **0.0000** — the published tripwire reproduces
  there. It is the hub's *unseen pair on a SEEN maze* REQUEST LAYER that destroys it.

## 10:15Z — declared NOT-RUNs (never blanks, never zeros)
NR1 Shapes3D incumbent cell at 3 seeds (seed 0 only exists) · NR2 the GPU compute pin
(4000/32768, $26.46, blocked on the Head's Mac Docker daemon) · NR3 w8 single-design RESULT rows
(0/3, falsified on val) · NR4 hub/maze-encoder-loop (contingency did not fire) · NR5 w9/maze
post-deletion composition with the repaired deletion on TEST (VAL seed 0 only, −0.0120).
Each carries a launch line in `TABLES.md`.

## close — outputs and hygiene
- Written: `TABLES.md`, `TABLES.json`, `FINDINGS.md`, `figures/` (5 PNG), `build_tables.py`,
  `verify_tables.py`, `LOG.md`, `START_UTC.txt`, `END_UTC.txt`, the bundle.
- ⛔ `$CHLU_CLAUDE_ROOT` was never read. The source register in `TABLES.json` is the complete list of
  files this spoke opened.
- ⛔ Nothing written outside `/c/chlu/outputs/w10-seeds-and-tables/` and my own worktree.
- ⛔ Zero scored runs. Zero background jobs. No box-wide kill (no process was killed at all).

## close+ — bundle and clean-clone regeneration proof
- Committed `scripts/w10_build_tables.py` and `scripts/w10_verify_tables.py` (purely additive, no
  frozen harness file touched) as `abe56d8` on `w10/seeds-and-tables`.
- `git bundle create ... w10/seeds-and-tables` (that branch only, ⛔ never `--all`).
  `git bundle verify` → *"is okay"*, *"records a complete history"*; `git bundle list-heads` shows
  exactly ONE ref, `refs/heads/w10/seeds-and-tables = abe56d8 = HEAD`. Size 5,115,630 B.
- ⭐ **Clean-clone test:** cloned the bundle into an empty directory and ran the CLONE's
  `scripts/w10_build_tables.py` with `W10_OUT_DIR` pointing at a throwaway dir.
  → figures **byte-identical** (sha256 diff: zero) and `TABLES.md` **identical** modulo the build
  timestamp line. The acceptance criterion is demonstrated end to end, not asserted.
