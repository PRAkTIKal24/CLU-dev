# w24 repro-package — advisor report

**Deliverable:** `nclu-iclr-code/` — an anonymous, standalone package (195 files, 46 MB incl. 45.7 MB of
frozen weights; 54,889 lines of Python), built mechanically from the source clone and the w15/w17/pw1/pw2
branches (tree-shaken to what the paper's entry points reach, renamed `chlu` → `nclu`, `CHLU_*` → `NCLU_*`,
programme vocabulary removed, comments dropped, docstrings kept only where neutral). Every location is an
`NCLU_*` variable with a default relative to the package root. No `.git`, no result, no trained
store/baseline checkpoint, no absolute path, no `authors` field; `LICENSE` reads "Licence to be added on
publication."

Companion files in this folder: `MAP.md` (every table/figure → origin → package entry), `scan-report.md`,
`proof/smoke.log` + `proof/smoke.log.progress` (proof C, all smokes), `proof/smoke_resmoke.log*` (the two
reports re-smoked after AMENDMENT-2), `proof/repro.log` (full reproductions + paper-cell comparison),
`proof/proofB_*` and `proof/repro_cmazet_raw.log` (earlier fresh-venv proofs), `NEEDS-ARTEFACTS.json`,
`LOG.md`, `tooling/` (build pipeline, scanner, positive control; never shipped).

## 1. Reproductions (fresh copy, fresh `uv venv`, `PYTHONPATH` unset, no environment script; seed 0 only — the only feature cache on this machine)

| paper cell (tex line) | printed | reproduced | verdict |
|---|---|---|---|
| Tab. t7 n-CLU K*=32, seed 0, all 109 combinations (l.554) | 47.495 | 47.495 | **MATCH** |
| Tab. t7 four landscape controls (l.567) | 0.000 | 0.000 ×4 | **MATCH** |
| Tab. t7 table floor seed 0 (l.565) | 8.229 | 8.229 | **MATCH** |
| K* footprint: address book / four-part total, seed 0 (l.647/649) | 1,060,864 / 2,144,178 B | same | **MATCH** |
| Tab. del item 1% seed 0: rows, before→after (paired), downdate−refit, statistic dev, statistic bytes (l.856, 871) | 1,311; 47.495→47.477 (−0.018); +0.000; 0; 967,304,800 | same | **MATCH** |
| Tab. del item seed 0: (a) max mode dev; (b) rendered rel dev | 2.34e-07; 5.83e-10 | 8.10e-08; 2.14e-09 | different values, both float roundoff |
| Tab. gentable-kstar K* seed 0 aggregate, bytes (l.1399) | 0.5718, 2,145,738 | same | **MATCH** |
| … controls removed / shuffled / random atoms (l.1400-1403) | 0.0556 / 0.1481 / 0.1458 | same | **MATCH** |
| … shuffled across wells (l.1402) | 0.1065 | 0.1273 | **MISMATCH** (the code path is byte-identical to the original; val gate also differs only here, 0.1266 vs 0.1217; cause not established — possibly the well-label order returned by the clustering) |
| Tab. cmazet-sym words+obs n-CLU seed 0 (l.2366) | 0.800 | 0.8003 | **MATCH** (bit-identical to the original run record, 0.8002628) |
| … highest landscape control, blind-sequential, word lookup (l.2369) | 0.415, 0.239, 0.000 | 0.4152, 0.2392, 0.000 | **MATCH** |
| … best matched baseline (l.2366) | 0.627 | 0.8003 (LSTM, 80,000 steps) | **MISMATCH** |
| … n-CLU − baseline (l.2366) | +0.173 | +0.000 | **MISMATCH** (follows) |
| Tab. cmazet per-step accuracy seed 0 (l.2263) | 0.930 | 0.9303 | **MATCH** |

Runtimes on this 10-core laptop CPU: partition 14 s; K*=32 build 90 s (address book 87 s); scoring the
n-CLU + 4 controls 95 s; item deletion 233 s (book refit 86 s, printed 5,427 s on the origin machine);
generation-table K* row 321 s; CMAZE-T words+observation TEST full run 219–298 s (origin: 5,135 s).

**The cross-machine finding, not softened.** Gradient-trained baselines do not reproduce across machines.
Evidence: I ran the ORIGINAL, unmodified runner from the source clone (`scripts/w19_symmetric.py`,
seed 0, BOTH, TEST) on this machine: every arm equals the package's run to the last digit
(n-CLU 0.8002628; rival@10x-steps LSTM 0.8002628; rival transformer 0.6780552; rival MLP 0.4875164 …),
while the original run record reads rival@10x-steps LSTM 0.6268068, rival transformer 0.4717477,
rival@10x-steps MLP 0.6123522. So the package is faithful and the difference is the machine. The
n-CLU, the four controls, the learned blind arms, the lookup floors and the planner columns are bit-exact
across machines; the matched baselines (and therefore every n-CLU − baseline margin on CMAZE-T) are not.
The same is expected of the Shapes3D baselines (not re-run here at full size; only the smoke fits).

## 2. Every table and figure → command

Full commands, with runtimes, are in the package README §5. Summary (status as in MAP.md):

| item | command (package) | status |
|---|---|---|
| Tab. t7 | `shapes_build_nclu.py`, `shapes_score_arms.py`, `shapes_fit_baseline.py`, `python -m nclu.experiments.baseline_suite --stage table`, `shapes_tables.py` | shipped; seed 0 reproduced |
| K* footprint table | `shapes_build_nclu.py` (info.json) | shipped; seed 0 reproduced |
| Tab. ladder | `shapes_ladder_earlier.py`, `shapes_ladder_nclu.py`, `shapes_fit_baseline*.py`, `shapes_score*.py`, `baseline_suite --stage learners`, `shapes_coverage_control*.py`, ladder tables | shipped; smokes pass (rung 2, seed 0) |
| Tab. del | `shapes_delete_nclu.py` | shipped; item row seed 0 reproduced |
| Tab. hole | `shapes_delete_nclu.py` (c), `shapes_refusal_comparator.py` | shipped |
| Tab. gentable-kstar | `shapes_generation_table.py --arm kstar` | shipped; seed 0 reproduced except across-wells |
| Tab. lace-poe | `shapes_fit_lace_poe.py`, `shapes_tables_lace_poe.py` | shipped |
| Tab. k5 | `shapes_build_nclu.py --K 5`, `shapes_score_arms.py` | shipped |
| Tab. kcurve / kcurve3, Fig. crossover | `shapes_sweep_k.py`, `shapes_sweep_k_seeds.py`, `shapes_sweep_k_report.py`, `shapes_sweep_k_figure.py` | shipped |
| Tab. impl | `python -m nclu.experiments.deletion_full_square`, `shapes_deletion_battery_candidates.py` | shipped |
| Tab. ablations | `python -m nclu.experiments.composition_cell` | shipped (pins smoke only) |
| Tab. rivals-del / rivals-del-seeds, Fig. deletion | `shapes_fit_baseline.py --delete-scope`, `shapes_score_arms.py`, `shapes_tables.py`, `figures/results_figures_b.py` | shipped |
| Tab. encloop, Tab. bard-gpu | `python -m nclu.experiments.encoder_loop` + `composition_cell` + `encoder_loop_report.py` + `encoder_loop_criteria.py` (README gives the sequence; GPU pins documented) | shipped; toy CPU smoke of warm and train; GPU leg not run |
| Tab. crr | `python -m nclu.experiments.retention_law`, `retention_law_criteria.py` | shipped (dev-stage smoke) |
| zero-dynamics check | `scripts/zero_dynamics_check.py`, `nclu.experiments.settle_dynamics` | shipped |
| Tab. cmazet | `cmazet_asymmetric.py`, `cmazet_select_config.py`, `cmazet_criteria_asymmetric.py`, `cmazet_tables_asymmetric.py` | shipped |
| Tab. cmazet-sym, Fig. maze-wm | `cmazet_symmetric.py` (+ `--clu_cfg scripts/configs/nclu_rel.json`), `cmazet_tables_*.py`, `cmazet_holdout_support.py`, `cmazet_content_diagnosis.py`, `figures/results_figures_b.py` | shipped; seed 0 reproduced (n-CLU) |
| Figs. task-shapes, task-cmazet | `nclu.task_assets.*`, `figures/task_figures.py` | shipped (CMAZE-T figure drawn in the smoke) |
| Figs. shapes-per-model, shapes-baseline-wins | `python -m nclu.render_shapes.build` | shipped (needs the Tab. t7 objects) |
| Figs. ladder, deletion | `figures/results_figures_b.py` | shipped |
| Figs. cmazet-families, -per-arm, -rel-per-arm, -door-tint | `figures/cmazet_extract*.py`, `figures/results_figures_b.py` | shipped (need the persisted CMAZE-T objects) |
| Tab. adapt, Fig. adapt | — | **left out** (below) |
| `\TODO` prose (l.126, 1307, 1329), declared not-runs | — | no numbers, no code |

## 3. Deliberately left out, and why

* **Tab. adapt and Fig. adapt (encoder adaptation).** Outside the Head's declared scope list (encoder
  training is not in it) and not named by AMENDMENT-1. The code exists (`HEAD:chlu/experiments/
  exp_w13_encoder_adapt.py` + `scripts/w13_score_leg.py`) and could be folded in the same way as the encoder
  loop now that the encoder ships. **Head decision.**
* The paper's physics appendix beyond the retention-law table and the zero-dynamics check (prior
  derivations, the "shorts" items, the ten requirements): scope says physics code only for those two.
* Retired comparators the paper no longer prints: the 16,384-row / 1,280-step baselines, `learned_1x`
  (ladder default arms changed to `store,controls`), the carried-book deletion comparison.
* Shell wrappers of the encoder-loop scoring (`w8_encoder_loop_{pipeline,score}.sh`, `w5_compose_field_leg.sh`,
  `w15_bar_d_*.sh`): they only set variables and call Python stages; their sequence is in the README.
* `peer_*` vendored copy and the three CMAZE-T runner variants: merged into one runner (`cmazet_symmetric.py`)
  whose three differences are additive prints/hooks with unchanged defaults; the zero-fit per-tuple replay
  script is not needed because the merged runner prints the per-tuple scores itself.
* Everything the cut sections of the draft used (CMAZE-O, CMAZE-2/2b, routes/read verb), the programme's
  registry/ledger/gate machinery, gates on external records (removed; see §6).

## 4. Frozen weights (measured) and decision

| file (per seed 0, 1, 2) | bytes | decision |
|---|---:|---|
| `weights/decoder/decoder_v1_s<s>.eqx` (frozen decoder, 1,516,227 params) | 6,070,076 | shipped (≤ 50 MB) |
| `weights/classifier/seed_<s>/external_classifier.eqx` (judge, 978,873 params) | 3,919,332 | shipped |
| `weights/encoder/shapes3d_recon/seed_<s>/encoder.eqx` (frozen encoder, 428,576 params) | 1,717,168 | shipped (AMENDMENT-1) |
| `…/probe.eqx` (factor probe, 633,049 params) | 2,535,268 | shipped |
| `…/decoder.eqx` (pretext decoder of the encoder run, 250,467 params) | 1,004,988 | shipped |
| frozen-encoder feature cache `features.npy` (1,966,080,128 B per seed) | 1.97 GB | NOT shipped (> 100 MB): sha256 per seed in README + `nclu/data_manifest.json`; loader expects `data/features/shapes3d_recon/seed_<s>/features.npy`; rebuildable with `baseline_suite --stage cache2` |

Total weights 45,740,496 B. No file in 50–100 MB. The original pickles embedded the old package's class
paths, so every network was converted to an array-only `.eqx` + a registry entry; each conversion was
proven by reloading into a fresh skeleton and asserting the parameter hash equals both the pickle's and the
run's manifest (15/15). The JSON manifests beside them were sanitised (absolute paths dropped, strings
passed through the same scrub).

## 5. Smokes (proof C: fresh copy + fresh venv; `proof/smoke.log`)

46 entry-point smokes in proof C; 44 exited 0. The two that exited 1 (`shapes_sweep_k_report.py`,
`shapes_coverage_control_report.py`) crashed on partial input (the three-seed aggregations with only seed 0
present). Both were fixed after AMENDMENT-2 to aggregate the seeds present and name the missing ones, and
re-smoked with their upstream steps in another fresh copy + venv: all ten steps exit 0
(`proof/smoke_resmoke.log.progress`). `proof/sha256_diff_proofC_vs_final.txt` shows that the final package
differs from proof C's copy in exactly those two report scripts, which the full reproductions do not
execute, so the reproductions were not re-run. Final count: 46 of 46 smokes exit 0. Smoke depth, stated plainly: figure builders other than the CMAZE-T task figure, the render
figures and `cmazet_extract*` are smoked by `--help` only (their inputs are full multi-seed runs);
`composition_cell` and `settle_dynamics` by their `pins` stage; `retention_law` by its `dev` stage.

## 6. What changed relative to the original code (functional changes only)

* Locations: every absolute path / `CHLU_*` variable → `nclu/paths.py` (`NCLU_DATA_ROOT`, `NCLU_FEATURES_ROOT`,
  `NCLU_WEIGHTS_DIR`, `NCLU_OUT`, `NCLU_ENCODER_ROOT`, `NCLU_DECODER_DIR`, `NCLU_BASELINES_ROOT`).
* Gates removed: scored-run ledger checks, "baselines ready" / pre-registration file checks, worktree-name
  import assertions (replaced by "the package resolves inside this folder").
* Weights: loaded from the `.eqx` registry (hash-checked); the feature-cache generation check became a
  presence + size check against `nclu/data_manifest.json`; a full-scope cache without a rows file uses rows
  0..n−1 (the recorded rows hash is exactly that of `arange(480000)`).
* Deletion script: deletes from this package's own K* object and scores "before" itself instead of reading
  an earlier deletion record; the carried-book comparison (retired) is dropped.
* Generation table K* launcher: the precondition on an unrelated experiment's refusal record is switched
  off (it gates other rows of that table); the store plan ships as `nclu/specs/store_plan_shapes3d_recon.json`.
* Full-square deletion battery: copies the condition's partition file into its output folder (the original
  run had it there).
* The three CMAZE-T runners merged (additive changes only); tables read all six seeds from one run folder.
* Two report scripts tolerate missing seeds (AMENDMENT-2).

## 7. Anonymity

`scan-report.md`: the real package scans EMPTY (0 hits over every file, name and binary structure); the
positive control catches every planted token (4 categories × body / file name / PDF metadata / PNG
metadata, plus compressed PNG zTXt, bytes appended to a weight file, an NPZ member name). The only
whitelist is the lowercase model "head" (the role word is matched case-sensitively).

## 8. Decisions the Head must make

1. **Licence text** (stub: "Licence to be added on publication.").
2. **Tab. adapt / Fig. adapt**: fold in the encoder-adaptation code (feasible now that the encoder ships) or
   keep it out.
3. **Reporting the baselines**: the CMAZE-T matched baselines (and every margin against them) do not
   reproduce across machines, even with the original code; the paper's baseline numbers are machine-specific.
4. **The across-wells control of the generation table** reproduces at 0.1273 against a printed 0.1065 (seed 0);
   the other four rows of that table match.
5. **Weights**: 15 frozen networks ship (45.7 MB); no file fell in the 50–100 MB band. The seed-1/2 feature
   caches are not staged and are left to `--stage cache2`.
6. Ambiguities I resolved (closest to what a reviewer needs): one merged CMAZE-T runner; `--nrows 32768` as
   the command for the 32,256-fit-row LACE/PoE rows; the generation table runs on the generation-2 cache
   (as the paper's caption states) with the encoder run beside it; baseline smoke fits at 2,048 rows.

## 9. Hub re-scan findings and fixes (AMENDMENT-3)

**What the Hub found that my scanner missed.** Review-item IDs `PW-1` ×18 and `PW1` ×4 (`PW1_BANNED`,
`PW1_TEXT`, `"PW-1 item 10: ..."` in figure code and `figures/data/*.json`), deletion scars (`(, 2026-09-25)` ×4,
`(, 2026-09-14)`, `-ruled 2026-09-25`), history fields (`retired`, `retired_by`, `rival_sets_retired`) and 21 ISO
development dates. My rule `\bpw[0-9]\b` could not match `PW-1` or `PW1_`, and nothing looked for the stub a
deletion leaves behind.

**Scanner (extended, nothing removed).** Added every rule the amendment lists (`PW-?[0-9]`, `(?<![a-z])pw[0-9]`,
`\bitem [0-9]+(\.[0-9]+)*\b`, `\(\s*,`, `(?<![A-Za-z])-ruled`, `\bruled\b`, ISO dates, `retired_by`), plus
`\bretired\b`, `AMENDMENT-?[0-9]` and review-round references `\bR[0-9]{1,3}(\.[0-9]+)?\b` (case-sensitive, so
the task's `raw r1` label is untouched). I also added a new category, "scrub scars". It catches the stubs a
deletion leaves: comma-led or empty parentheses, a space before `,`/`;`, `.:`, `on -<n>`, a doubled article, an
orphan possessive `'s`, `( ii)`, `PROTOCOL-§`, `ABC-)`, the stub docstring `init  .` and a dangling `spent in;`.
The positive control now also plants `PW-1`, `PW1_X`, `(, 2026-09-25)` and `(, )`: **23/23 caught.**

**Before → after.** The final scanner on the proof-C copy (the harvested package) gives **216 hits**. On the rebuilt
package it gives **0 hits** (196 files including MANIFEST.md). Largest classes before: review rounds 77 (42 of them are the
module alias `gen_deletion as R7`, and 4 are functional labels `R0`/`R2`/`R3`), `PW-?[0-9]` 28, orphan `'s` 17, dates 16,
`item <n>` 14, `retired` 13, `pw[0-9]` 11, `init  .` 10, `(,` 9, `retired_by` 6. The per-rule table is in
`scan-report.md` §6. The extended scanner also exposed scars from earlier rounds that the Hub had not listed
(e.g. `"The 's seven-row taxonomy"` from "The Head's…", and `"words are n in"` from "spoken in", where `spoke` had been cut out of the word).

**Fixes, all in the build transform, then a full rebuild (no hand edits):**
* Review-item, review-round and amendment references are dropped together with the preposition that introduced them.
  A possessive reference becomes "the": "R8's declared embedding" → "the declared embedding".
* Identifier renames: `PW1_BANNED` → `FIGURE_TEXT_BANNED`, `PW1_TEXT` → `FIGURE_TEXT_SEEN`,
  `check_vocab_pw1` → `check_figure_vocabulary`, figure stems `rc_pw1_shapes_*` → `shapes_*`, alias `R7` → `GD`.
* `retired_byte_budget` is **functional** (688,128 B; the rival budget notes read it), so it is kept and renamed
  `prior_byte_budget`. `retired_rows` is **not read** by any code, so it is dropped.
* Figure data: `retired`, `retired_by` and `_doc` are dropped; no code reads them. The status value `retired` becomes
  `not_drawn`. The output key `rival_sets_retired` becomes `rival_sets_not_drawn`; `results_figures_b.py` computes it
  from the status.
* Development dates are removed from strings, docstrings and data. A string that is only a timestamp becomes "not recorded".
* Scar repair (`descar`) runs after every deletion and never touches the edges of an f-string fragment. Words that
  contain a banned token are replaced whole (`spoken` → `expressed`, `pre-registered` → `declared in advance`).
* Before the proof, my own review of the diff caught two errors that an early draft of these rules had introduced.
  One glued an f-string field to the next word (`{WHITENING_D_LIMIT}full`). The other turned the regex look-behind
  `(?<!n-)` in the figure-vocabulary check into `(?<!n)`. Both were fixed in the transform. A masked-AST comparison
  of every `.py` file against the proof-C copy now shows only the renames above and the two AMENDMENT-2 report fixes.

**Proof on the rebuilt package.** The sha256 diff against the proof-C copy shows 61 files changed (57 `.py`, 4 JSON)
and 134 unchanged (`proof/sha256_diff_proofC_vs_amend3.txt`). The changed files include `nclu/config.py`,
`gen_landscape.py`, `factor_wells.py` and the figure code, which every entry point executes, so I re-ran the whole
proof (proof E: fresh copy, fresh `uv venv`, `PYTHONPATH` unset, PID-based wait). **All 46 smokes exited rc=0**
(`proof/amend3_smoke.log*`). Both full reproductions were re-run (`proof/amend3_repro.log`). **Every number is
identical to proof C**: Shapes3D 47.495 with its four controls at 0.000, the table floor 8.229, the footprint, the
deletion row, the generation table K* row, and every CMAZE-T arm. The paper comparison in `proof/repro.log`
therefore stands unchanged, including the cross-machine baseline finding (§1, §8.3) and the across-wells mismatch (§8.4).
The proof-E copy is byte-identical to the shipped package, except for MANIFEST.md, which was regenerated afterwards
(196 files).
