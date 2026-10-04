# FIGURES.md

<!-- PW1-SECTION-START -->
## ⭐ PW-1 figure install (pw1-draft-figures, 2026-09-25): the current state. Read this section first.

Built by `w21_figures/build.sh` at `f571b2b7561347ec7498611a77a1c5ab48f990ba` (two builds byte-identical, each equal to the figure spokes' own builds), installed into `w17-paper/figures/` and `paper-visuals/`; rendered figures from pw1 render-shapes' build. Captions and slots in both drafts by pw1-draft-figures. The full per-file index is `README.md`.

| figure | item | code branch | what changed | label, figure no. | installed SHA-256 (pdf) |
|---|---|---|---|---|---|
| `fig_maze_conditions` | PW-1 items 1, 8 | `pw1/plot-maze` | redrawn: four rows (words full, observation, words + observation for n-CLU and n-CLU-REL), every seed drawn, no claim labels, no ceiling, (c) relational comparison only; new caption | Figure 9, `fig:maze-wm` | `3896e8128efb912bcdc9ec8f22909d7de67d3930671fe22f1a0f6a10eeee47e6` |
| `fig_cmazet_families` | PW-1 item 16 | `pw1/plot-maze` | no bottom paragraph (read once in Sec. maze-wm); plain names; + most-baselines-wrong row and a door success row (n-CLU-REL run, seed 2), drawn by rule, seed 16 | Figure 10, `fig:cmazet-families` | `b774ac729782480d5c279a62910f3259dd4b87ece63aab7bf9bc174f891e17ae` |
| `fig_cmazet_per_arm` | PW-1 item 16 | `pw1/plot-maze` | no bottom paragraph; plain names; no door success exists for the n-CLU (stated in the caption) | Figure 11, `fig:cmazet-per-arm` | `744e643df043307e89844f011d2195dd1ac55b4daaad05ce0ba6216a35c278a6` |
| `fig_cmazet_rel_per_arm` | PW-1 item 16 | `pw1/plot-maze` | no bottom paragraph; plain names; n-CLU-REL success row is a door success drawn by rule | Figure 12, `fig:cmazet-rel-per-arm` | `22454719c5f85e3ad0d4d443d0402019c234c5716aedd980ba66212e9ad31d23` |
| `fig_cmazet_door_then_tint` | PW-1 item 16 | `pw1/plot-maze` | no bottom paragraph; + three n-CLU-REL door successes (seed 2) drawn by rule | Figure 13, `fig:cmazet-door-tint` | `ea36760fe82cfc2d38c0d080a25cb1c24979dcc08fc780de4dc66bbeeba193b8` |
| `fig_ladder` | PW-1 item 10 | `pw1/plot-shapes` | single data-matched panel, no title, legend right, no line-end labels, new names and y label; new caption | Figure 5, `fig:ladder` | `949d5b6d1ff2fac3048bf203c22326c36d66d8fed43e0e67a2aae93d09685f2e` |
| `fig_crossover` | PW-1 item 11 | `pw1/plot-shapes` | axis labels, panel (b) title "Memory footprint vs K", legend n-CLU; new caption | Figure 6, `fig:crossover` | `0dfda2510acbce7f4335b8c0a1c5f068e80a93f780a1673a60e7ed577bf24c36` |
| `fig_deletion` | PW-1 items 12, 13 | `pw1/plot-shapes` | only former (e), (f) remain as (a), (b), retitled; red sphere; the removed panels' numbers are in Tab. del and one sentence | Figure 7, `fig:deletion` | `8faeaf9fa28f05d86a1a201b29b8f8cfadb9cd9d4a1e46cc3cd07092f5c73773` |
| `fig_encoder_adapt` | PW-1 items 2, 3 | `pw1/plot-shapes` | y label in the new vocabulary; caption matches it | Figure 8, `fig:adapt` | `3c0a924da28bf664d96a9fd49970e3a53257e345cc7463626729d31a71bbacdb` |
| `rendered/rc_pw1_shapes_per_model` | PW-1 item 9 | `pw1/render-shapes` | NEW: per-model success/failure grid against the w20 data-matched baselines, seed 0; Sec. shapes-t7 | Figure 3, `fig:shapes-per-model` | `f9dfc17eacf0e6add220447fe7d8467c1e24f0ac5e9a6c273d7b8d81bfb8d0e0` |
| `rendered/rc_pw1_shapes_baseline_wins` | PW-1 item 9 | `pw1/render-shapes` | NEW: baseline succeeds, n-CLU fails, one row per baseline, seed 0; Sec. shapes-t7 | Figure 4, `fig:shapes-baseline-wins` | `18661bc6d18ee5f5901a24afef8a81b59dc0e4718da28ce918caf8e8b90383b4` |

**Retired (moved into `retired-pw1/`, never deleted):** `rendered/retired-pw1/rc_deletion_red_sphere`, `rendered/retired-pw1/rc_set1_clu_composes_rivals_fail`, `rendered/retired-pw1/rc_set2_all_arms_succeed`, `rendered/retired-pw1/rc_set3_all_arms_fail`, `rendered/retired-pw1/rc_w19_cmazet_deletion`, `rendered/retired-pw1/rc_w19_cmazet_door_then_tint`, `rendered/retired-pw1/rc_w19_cmazet_families`, `rendered/retired-pw1/rc_w19_t7_per_arm_success_failure`, `rendered/retired-pw1/rc_w19_t7_rival_succeeds_clu_fails_seed1`, `rendered/retired-pw1/rc_w19_t7_rival_succeeds_clu_fails`, `retired-pw1/fig_cmazet_deletion`, `retired-pw1/fig_headline_steps`, `retired-pw1/fig_maze_readverb`, `retired-pw1/fig_maze_rel_tuples`, `retired-pw1/fig_maze_requestlayer`, `retired-pw1/fig_maze_wm`.

Unchanged and placed: `fig_task_shapes` (w19 mirror), `fig_task_cmazet` (built, unchanged bytes).

Sections below are history (w22, w21, w17); where they disagree with this section or README.md, this section wins.
<!-- PW1-SECTION-END -->

## ⭐ w22 `figures-redraw` pass 3 (2026-09-25): the current state. Read this section first.

**Zero scored runs; nothing written to any ledger; no PREREG.** Build: `bash build.sh` (unchanged
recipe + `fig_task_cmazet` and its glyph gate), twice, byte-identical, installed into both
`w17-paper/figures/` and `paper-visuals/`, re-hashed at each.

| stem | change | source |
|---|---|---|
| `fig_cmazet_families`, `_per_arm`, `_door_then_tint`, `_deletion` | ⭐ **fact strip under every $s_{t+1}$** (engine column: the truth): line 1 `legal`/**`illegal`**; line 2 move (↑→↓←, · = stay), key event (+/− colour), pocket after the move (disc, white = empty). An illegal emission shows only its pocket (the judge scores nothing else then). The ✓/✗ is the judge's verdict on all four facts. Drawn state and disc UNCHANGED. `_per_arm` now 3 blocks (page fit). | `cmazet_panels.fact_strip`/`facts_shown`; `data/cmazet_items.json` |
| `fig_cmazet_rel_per_arm` (NEW, `fig:cmazet-rel-per-arm`) | CLU-REL-A10 per-arm success/failure, 19 arms incl. six `+rel` diagnostics; same rule/seed as `_per_arm` | gate 2: `cmazet_extract_rel.py` → `data/cmazet_items_rel.json`; 132/132 arm scores float== run JSONs, seeds 0–5 (`data/cmazet_items_rel.REPRO.json`) |
| `fig_task_cmazet` (AMENDMENT-1) | every maze cell drawn by `cmazet_panels.draw_state` (via `draw_grid`): agent = black disc, centre = pocket colour (white = empty); keys diamonds, goal a cross; c1–c4 = the panels' `MAZE_RGB`; **no t / budget line**; digits overlaid; leader order = fewest crossings (0) | `w19_task_figures/make_task_figures.py --only cmazet`; `GLYPH-IDENTITY.json` (`verify_glyph.py`) |

**Same-picture / opposite-mark pairs** (`cmazet_ambiguity.py` → `data/CMAZET-AMBIGUITY.json`; picture = rendered pixels):

| items | arms | opposite-mark pairs | same picture BEFORE | … legality only visible | Head's rule (differ only in legal bit) | AFTER (pixels + strip) |
|---|---|---|---|---|---|---|
| A1 (w19), seeds 0–2 | all 16 | 92,628 | 33,412 | 18,035 | 3,740 | **0** |
| A1 (w19), seeds 0–2 | 13 drawn | 63,624 | 22,340 | 12,624 | 1,261 | **0** |
| CLU-REL-A10, seeds 0–5 | all 22 | 373,583 | 154,671 | 107,148 | 19,915 | **0** |
| CLU-REL-A10, seeds 0–5 | 19 drawn | 295,294 | 124,450 | 88,384 | 13,115 | **0** |

⚠ Legality alone does not reach 0: the emitted pocket-next and incoherent key events are also invisible in the state; the strip prints all scored facts, so the count is 0 by construction and measured 0.

⚠ **Palette change (task figure only):** `fig_task_cmazet`'s c1–c4 were dataviz slots 1/2/3/7 (blue, orange, aqua, violet), which disagreed with every result panel (c1 red there). They are now the panels' `MAZE_RGB` (red, blue, amber, purple), so a colour means the same key everywhere. Identity also carried by digits.
⚠ **`MAZE_RGB` c4 re-stepped** (0.48, 0.25, 0.63) → (0.61, 0.25, 0.63): the pass-1 4-set FAILED the dataviz normal-vision floor (blue↔purple ΔE 13.7 < 15; CVD 7.1). New set, `validate_palette.py --pairs all`: CVD ΔE 8.2, normal 17.8, ALL CHECKS PASS; amber stays below 3:1 on white (WARN; relieved by digits in the task figure and by the colour numbers in every panel row label).


## ⭐ w22 `figures-redraw` pass 2 (2026-09-24): the current state. Read this section first.

**Zero scored runs; nothing written to any ledger.** Build: `bash build.sh` = `make_figures.py` then
`make_figures_w22.py` (which imports `make_figures_w21.py` as a library; `maze_six.py` reads the
six-seed CMAZE-T artefacts), twice, SHA256-compared (`SHA256-w22-build{1,2}.txt`), installed
byte-identically into **both** `w17-paper/figures/` and `outputs/paper-visuals/`.

| stem | what it shows | source (every value read, never typed) |
|---|---|---|
| `fig_ladder` | store vs **three** rival sets, one panel each; **pool rows AND optimiser steps in every panel title**: (a) 32,256 rows · 5,040 steps (spoke-A `@ref`), (b) 32,768 rows · 2,560 steps (data-matched MLP), (c) **131,072 rows · 20,480/204,800 steps** (w20, P1-carried fit per family × seed; filled dot = seed carried at 204,800) | `data/ladder_rival_sets.json`; w20 `SHAPES-MATCHED.json ladder.*`; steps asserted per rung × seed × family |
| `fig_headline_steps` (NEW, `fig:headline-steps`) | rows 3–4 TEST: rival score against optimiser steps at 16,384 / 32,768 / 131,072 rows — the operative variable was steps, not data | `SHAPES-MATCHED.json rows34.arms`; each fit's `declared.json`/`tenx.json` `optimizer_steps`, `n_fit_rows` |
| `fig_maze_conditions` | (a) condition table + CLU-REL-A10 row (Head's rule, seeds 0–2); (b) A1 and (c) CLU-REL-A10, **six seeds**, per-seed **ceiling** bars, **hard seeds 1, 4 shaded** (A1 ceiling < cut 0.6407), best matched rival (open ▲), best **+rel** rival (filled ▲), highest control (×); every seed where a rival/control reaches the CLU is **marked by rule** (seed 4: control = CLU; seed 5: rival = CLU; A1 seeds 1, 5: rival > CLU) | per-seed run JSONs `ROWS.nway`, asserted float-equal to `REL-NEWSEEDS.json nway_summary`; `MAZE-T-SEED4.json classification` |
| `fig_maze_rel_tuples` (NEW, `fig:maze-rel-tuples`) | CLU-REL-A10, every held-out n-way tuple × 6 seeds: CLU, best matched rival, best +rel, highest control; "= CLU on k/n" per row (seed 4: control and +rel 16/16; seed 5: rival and +rel 16/16) | `w21-rel-newseeds/replay/BOTH_test_seed*_PERTUPLE.json` (stored `[n, correct]`); objects hash-identical to the staged ones |
| `fig_deletion` | the **four measurements** (w20, book refit): (a) store = refit-without, (b) book = refit-without, (c) refusal display, (d) before→after with **the random-1 % item control beside the conjunction**, (e) paired Δ + CI, (f) CLU after − each rival retrained without | `SHAPES-MATCHED.json deletion_matched.per_scope.*` |
| `fig_crossover`, `fig_cmazet_*`, others | unchanged from pass 1 | — |

**Colour roles (dataviz, validated palette unchanged):** CLU blue; per-seed "best matched rival" is a
*role* (its family varies by seed) → slot 2 orange; **open = without the relational request columns,
filled = with them**, for both the CLU (A1 vs REL) and the rival (matched vs +rel); controls grey ×;
ceiling ink bar. Families keep their pass-1 hues wherever a family is the entity.

**Pending slots (data, not code):**
- `data/ladder_rival_sets.json` → `ladder_rivals_32k` (32,768 rows; still running on forgis-14). Set
  `status: landed`, `source`, `reader`, `arms`, `steps`, `steps_check`; rebuild. Tested with a stand-in
  4th set: the figure becomes a 2 × 3 grid, no code change.
- ~~`data/cmazet_panels.json` → `fig_cmazet_rel_per_arm`~~ LANDED in pass 3 (see above). Was: gate 1 PASSED
  (192/192 objects), gate 2 cannot run — the REL pickles need module `w21_clu`, absent on batmac
  (`NEEDS-ARTEFACTS.json` in the w22 outputs).


## ⭐ w21 `figures-redraw` (2026-09-24): the current state. Read this section first.

**No figure carries an in-image banner.** Placeholder status lives only in PROVENANCE.json and the
draft's placeholder index. `make_figures.py`'s `tag_placeholder` is a no-op now. The w17 text
below is kept as history; its banner rule and its placeholder list are **superseded**.

**Rebuild:** `bash build.sh` from this directory. It runs `make_figures.py`, then
`make_figures_w21.py` (with `--panels data/cmazet_items.json` when that file is present), twice
into scratch dirs. It diffs SHA-256 over every output and installs only if both builds match.
Proof files: `SHA256-w21-build1.txt` = `SHA256-w21-build2.txt`. The earlier `SHA256-build*.txt`
and `SHA256-maze-w18-*.txt` are kept as history. `PROVENANCE.json` is merged, never dropped: each
entry carries its sources and the **SHA-256 of every source file**.

**Rows, rival sets and panel sets are DATA** (`data/`). Adding one means editing a JSON file and
rebuilding. It needs no code change.
- `data/ladder_rival_sets.json`: one entry per rival column set, and one panel per landed set.
  Each set's pool size is asserted from its own artefact. Landed: the 32,256-row
  combination-split set and the data-matched MLP (32,768 rows). **Not landed on this machine:**
  the 131,072-row set (w20 `shapes-data-matched`) and `spoke/ladder-rivals-32k`. To add one, set
  `status: landed` and give `source`, `reader`, `arms`, `pool_source` and `pool_path`, then add a
  caption line to the draft.
- `data/maze_conditions.json`: the rows of the condition table, top to bottom. The
  `words_plus_observation_improved` row waits for `w21-maze-t-clu-improve`: set `status: landed`
  and give its BARS artefact and schema.
- `data/cmazet_panels.json`: one entry per Part-C panel set, and one column per arm. An improved
  CLU is added as an arm entry, plus its own per-item re-read made by `cmazet_extract.py`, which
  must pass the same reproduction gate.

| figure | shows | source artefacts | seeds |
|---|---|---|---|
| `fig_crossover` | (a) val composition vs K: mean ± 1 seed sd, each seed, and the highest control; (b) four-part ledger (seed mean, stacked, sd on the total) | `w15-crossover-seeds/CROSSOVER-3SEED.json` | 3 at **every** K |
| `fig_ladder` | store vs each rival set, one panel per set, with the table floor | `w12b-nr6-and-kstar/KSTAR-LADDER.json`, `w12-composition-ladder/LADDER.csv`, pool files | 3 |
| `fig_deletion` | (a) survivor composition before/after for the store and 4 rivals × 3 scopes; (b) seconds; (c) rows read | `w15-single-design-seeds/T7-3SEEDS.json` | 3 |
| `fig_maze_conditions` | CMAZE-T condition table: per-seed CLU and R\*, the Head's-rule label, no margin bar | BARS JSONs listed in `data/maze_conditions.json` | 3 per row |
| `fig_cmazet_families` / `_door_then_tint` / `_per_arm` / `_deletion` | every arm's generated s_{t+1} and the judge's mark | persisted w19 BOTH TEST objects → `data/cmazet_items.json` (re-read gate: `data/cmazet_items.REPRO.json`) | families: seed 0 (rule\_a from seed 2); others: seed 0 |
| `fig_encoder_adapt`, `fig_maze_readverb`, `fig_maze_requestlayer` | unchanged w17 figures, rebuilt without a banner (bytes equal the w18 installs) | as below | as below |

**Maze labels (the Head's rule, computed by one function from each BARS artefact):**
- Rivals on the observation: **ASYMMETRIC** (no claim label).
- Words, main effects (VAL): **LOSS**.
- Words, full design (TEST): **LOSS**. This equals the artefact's own `HEAD_LABEL` (asserted).
- Observation (VAL): **COMPARABLE**. |Δ̄| = 0.1053 ≤ σ = 0.1068. ⚠ The brief's table said LOSS;
  the rule applied to the artefact says COMPARABLE.
- Words + observation (TEST): **WIN** against R\* = LSTM at 80,000 steps, although w19's own
  pre-registered margin bar B1 failed.

**Part C gate:** the persisted objects were hash-verified (132/132, with MANIFEST, `.sha256` and
the recomputed `param_sha256`). They were then re-read on batmac. The BOTH TEST n-way reads are
0.800 / 0.566 / 0.808 (mean 0.7246), with controls 0.000 / 0.225 / 0.244 / 0.166. Every one of the
16 arms, and every deletion row, equals the scored JSON floats exactly on all 3 seeds.

**Palette:** these are the w17 arm colours, the dataviz reference slots. They were validated with
`validate_palette.py`, a Python twin of the skill's JS validator, because this box has no node.
The port reproduces the skill's documented 9.1 / 19.6. The 5-hue ladder set, checked over all
pairs, gives CVD WARN (6.1) and normal-vision PASS (15.6). Secondary encoding is therefore
applied: a distinct marker per family plus direct end labels, and the paper's tables serve as the
table view. Output is print PDF on white, light mode only.

---

## w17 `plots` (history; banner rule and placeholder list superseded by w21 above)

Owner: `w17/plots`. This directory (`figures/`) is owned by `plots`; ⛔ `figures/rendered/` belongs to
`rendered-compositions` and is never written by this spoke.

**Evidence base:** `~/clu/evidence/outputs/` (read-only). **No w15 or w16 artefact is read by any
figure** (AMENDMENT 1). Every plotted value is read from the files listed below by
`make_figures.py`; nothing is typed in, rounded or recalled. Machine-readable provenance:
`PROVENANCE.json` (written by the same build).

**Rebuild:** `bash build.sh` (from this directory, or from the `w17/plots` branch at
`w17_plots/build.sh`). It builds twice into scratch dirs, diffs the SHA-256 of every output and
refuses to install unless they match. Python 3.12.14 · matplotlib 3.10.8 · numpy 2.4.1 ·
backend Agg · font DejaVu Sans (matplotlib-bundled, TrueType embedded, `pdf.fonttype 42`).
Proof: `SHA256-build1.txt` = `SHA256-build2.txt` (see `~/clu/outputs/w17-plots/FINDINGS.md`).

## One arm, one name, one colour — in every figure

| arm | name on every figure | colour | marker |
|---|---|---|---|
| our object | **CLU** | blue `#2a78d6` | ● |
| 1× byte-matched learner | **Learned 1× (four-part total)** | orange `#eb6834` | ■ |
| MLP rival | **MLP** | aqua `#1baf7a` | ▲ |
| LSTM rival | **LSTM** | yellow `#eda100` | ▼ |
| Transformer rival | **Transformer** | magenta `#e87ba4` | ◆ |
| continuous-MLP rival | **Continuous MLP** | green `#008300` | ✚ |
| lookup table | **Lookup table (floor)** — a printed floor, never a rival | grey `#6b6a66`, dashed | × |
| charter-C controls | the control's own name | light grey `#b5b4ae` | × |
| CLU encoder configurations (fig 4 only) | **frozen encoder / adapter / anchor** | blue / violet `#4a3aa7` / red `#e34948` | ● ■ ▲ |

Colours are slots 1–8 of the dataviz reference palette in its documented (validated) order;
every series also carries a distinct marker or hatch, so identity is never colour-alone. The
CLU's own byte-ledger parts use one hue (the CLU's blue), light → dark. The maze request-layer
audit (fig 5b) shows **environments, not arms**, so it uses neutral greys and never an arm colour.

## ⛔ Byte convention (COMMON: "use ONE convention throughout, or print the bracket")

**ONE convention in every figure: the crossover hub's FOUR-PART honest ledger** — atoms +
covariance/count + address book + coupling/hole table, **summed, never netted** — with
524,288 B drawn only as a **reference line / reference-pin label**. The K = 5 full-square-priced
1× opponent (3,177,896–3,241,076 params, the row the 938,413–1,710,324 bracket was about) is
**not drawn anywhere**. The 524,288 B rivals are the read-path-matched naive rivals (~130 k
params). The w12 `pca_tensor` / `DowndatableStore` convention does not appear.

## The figures

### 1. `fig_crossover.{pdf,png}` — ⚠ PLACEHOLDER
(a) val composition vs K (wells); (b) four-part byte ledger vs K, stacked by part.
- **Source:** `w12-single-design-crossover/CROSSOVER-val.json` (`per_K.*.val_composition_per_seed`,
  `val_composition_mean`, `ledger_four_part.*`, `control_gate_at_K_star`).
- **Seeds:** n = 3 (0, 1, 2) at K = 5 and K = 32; **n = 1 (seed 0) at K = 8…28** — printed in
  the legend. The "3-seed band" the task asks for exists only at K = 5 and K = 32 in w≤14 evidence.
- ⚠ The artefact's first key is `SMOKE_NOT_A_CLAIM: true`. The plotted byte column is
  `honest_total_bytes` and its four parts (asserted to sum to it), **not** the parts-1-2-4
  column the w13 advisor report §5 flags as the Hub's error.
- **Awaiting:** w15 `crossover-seeds` (3 seeds at every K). ⛔ It may confirm, narrow, or
  overturn the no-crossover reading; the figure presumes nothing.

### 2. `fig_ladder.{pdf,png}` — ⚠ PLACEHOLDER
(a) composition vs n per arm at K\* = 32, mean line + min–max seed band, per-corner ceiling
dotted; (b) margin CLU − Learned 1×, per seed + mean + band, 0 and the declared +1.0 margin.
- **Sources:** `w12b-nr6-and-kstar/KSTAR-LADDER.json` (`by_rung.*.kstar_K32.{store,learned_1x,margin}.per_seed/mean`,
  `legs.*.kstar_K32.*.ceiling_per_corner`) and `w12-composition-ladder/LADDER.csv`
  (`rival_{mlp,lstm,transformer,cont_mlp,table}` rows, `mean_distinct_valid_novel_tuples`, seeds 0–2;
  these 524,288 B rivals do not depend on K).
- **Seeds:** 3 on every rung.
- ⛔ **The n-axis is labelled for what it is:** "n ≥ 3: the same 120 corners" — for n ≥ 3 it is
  a TRAINING-DATA axis; only n = 2 vs n = 6 is a depth contrast. The draft's caption must carry
  the Head's inference note; the figure carries it on the axis.
- ⛔ **Panel (b) is "margin vs Learned 1×", NOT "vs best rival".** The build checks whether
  Learned 1× is the strongest learned arm at every (rung, seed): **it is not** — at n = 3,
  seed 0 the 524,288 B LSTM and at n = 3, seed 2 the 524,288 B Transformer score above it.
  `\TODO{no artefact for the K* margin against the best of ALL learned arms at n = 3}` — the
  KSTAR-LADDER artefact prints margins against Learned 1× only, and this spoke does not
  re-derive a margin.
- **Awaiting:** wave-18 regeneration (Amendment 1 names the ladder). No w15/w16 ladder row is read.

### 3. `fig_deletion.{pdf,png}` — ⚠ PLACEHOLDER
(a) CLU composition before vs after deletion, per scope, TEST, K\* = 32; (b) adaptation seconds
(log); (c) data rows read — CLU downdate, CLU refit-without (the exact reference) and each rival's
retrain-without.
- **Sources:** `w12-single-design-crossover/deletion/test_seed0_K32/deletion.json`
  (`scopes.{value,conjunction_live,item}.{before_same_corners,after}`, `scopes.value.our_cost_line`,
  `scopes.value.exactness_store_scoped.refit_without_cost_line`, `scopes.value.receipt.n_before`);
  `w12-deletion-baselines/comparators/seed_0/{mlp,cont_mlp,transformer,lstm}__retrain_without__value-object_hue-0.rival.json`
  (`cost_line.{seconds,data_rows_read}`, `deletion_counts.n_fit_before`).
- **Seeds:** **1 (seed 0)** for the CLU; rival cost lines are seed 0 (rows read and bytes are
  seed-invariant per the spoke's own FINDINGS §4).
- ⛔ Pools differ and are printed on the axis: CLU 131,072-row pool, rivals 16,384-row pool.
- ⛔ Language carried into the notes: exactness is **store-scoped**; frozen encoder/decoder and the
  address book are **outside the deletable object**; adaptation is **light and data-free** — never
  "no changes". The vacuous red ∧ cube scope is **not plotted** (it deletes 0 rows).
- Rival post-deletion composition on the scored corners: **never measured** in w≤14 evidence
  (DECLARED NOT-RUN there), so (a) is CLU-only. `\TODO{no artefact for rival post-deletion composition on the scored corners}`
- **Awaiting:** w15 `single-design-seeds` (deletion rows at seeds 1–2). ⛔ Those may reproduce
  or break the seed-0 reading.

### 4. `fig_encoder_adapt.{pdf,png}` — FINAL (not a placeholder)
Per-seed margin (single design − best learned baseline, `mlp_180k`) for frozen → adapter and
frozen → anchor, TEST, K\* = 32.
- **Source:** `w13-encoder-adapt/CROSS-ARM.json` (`MARGIN.{adapter,anchor}.per_seed.*`).
- **Seeds:** 3 seeds × 3 configurations = 9 arm-runs (3 SEEDS, never "9 seeds").
- ⛔ Opponent is `mlp_180k`, **not** Learned 1× — never merged with fig 2.
- Needs no w15/w16 artefact.

### 5a. `fig_maze_readverb.{pdf,png}` — ⚠ PLACEHOLDER (maze)
Validity under the exact judge, AVG (latent mean) vs MERGE (plan-space verb), per arm, on the
`unseen_pair` and `n-way` rows; per-seed dots; pipeline-ceiling seed range shaded.
- **Sources:** `w14-read-verb/runs/scored1_seed0_test.json`, `scored2_seed1_test.json`,
  `scored3_seed2_test.json` (`table.{unseen_pair,nway}.<arm>.{AVG,MERGE_target_wins}.validity`,
  `pipeline_ceiling_rows.*.validity`). `MERGE_target_wins` is the arm per `w14-read-verb/TABLES.md`
  (verified by inspection this session: the JSON value 0.51256… matches the table's 0.5126 for CLU unseen_pair seed 0).
- **Environment:** CMAZE-2b only. **Seeds:** 3.
- ⛔ This is an **INTERFACE (verb)** result: the same merge lifts the lookup floor more than the
  CLU and the charter-C controls do not collapse ⇒ no store row. The lookup floor at ~0.97 is the
  **within-maze** floor that cannot transfer across mazes.
- **Awaiting:** w16 maze-wm rows (composition, rollout validity, deletion). ⛔ Each may land as a
  declared negative.

### 5b. `fig_maze_requestlayer.{pdf,png}` — ⚠ PLACEHOLDER (maze, APPENDIX)
(a) keys per optimal route, before vs after the request-layer fix; (b) scripted-planner validity
at the opt + 10 % budget.
- **Sources:** `w14-request-layer-fix/artefacts/measure_before_0.json`,
  `measure_after_{0,1,2}.json` (`key_count_histogram`, `all_pairs_slack10.blind.*.validity`).
- **Seeds:** before = seed 0 only (the only before-artefact on disk); after = 3.
- ⚠ **Judgement call, flagged for the Hub/referee:** this figure shows the CMAZE-2 and CMAZE-2b
  request layers side by side **because the figure is the fix itself** — an environment audit
  with no learned arm and no score compared across environments. If the Hub reads COMMON's
  "never in one table" rule as covering figures, drop panel (a)'s before bars and (b)'s before
  bars, or drop the figure (appendix only).
- **Awaiting:** w16 maze-wm (CMAZE-T) rows supersede the maze section; may be dropped or re-scoped.

## Placeholder list (for wave 18 `placeholder-fill`)

| figure | placeholder? | w15/w16 artefact it awaits | may land as a negative? |
|---|---|---|---|
| fig_crossover | **YES** | w15 `crossover-seeds` | yes — a crossover may appear or the flat reading may narrow |
| fig_ladder | **YES** | wave-18 regeneration (Amendment 1) | yes |
| fig_deletion | **YES** | w15 `single-design-seeds` (deletion rows, seeds 1–2) | yes |
| fig_encoder_adapt | no | — | — |
| fig_maze_readverb | **YES** | w16 maze-wm composition / rollout / deletion | yes — every w16 row may be a declared negative |
| fig_maze_requestlayer | **YES** (appendix) | w16 CMAZE-T harness | may be dropped |

Every placeholder figure carries the red tag **"PLACEHOLDER · w≤14 evidence · wave 18
regenerates"** above its panels. ⛔ The tag must be removed only by the regenerating build, never
by hand.

## LaTeX include block (paths relative to the draft; source comments travel with the figure)

```latex
% SOURCE: w12-single-design-crossover/CROSSOVER-val.json  (PLACEHOLDER: w15 crossover-seeds)
% BYTES: four-part honest ledger, summed never netted; 524,288 B is a reference line only
\includegraphics[width=\linewidth]{figures/fig_crossover.pdf}

% SOURCE: w12b-nr6-and-kstar/KSTAR-LADDER.json ; w12-composition-ladder/LADDER.csv  (PLACEHOLDER)
% BYTES: Learned 1x at the four-part K* total; rivals at the 524,288 B reference pin
% n >= 3 rungs share the identical 120 corners: a TRAINING-DATA axis
\includegraphics[width=\linewidth]{figures/fig_ladder.pdf}

% SOURCE: w12-single-design-crossover/deletion/test_seed0_K32/deletion.json ;
%         w12-deletion-baselines/comparators/seed_0/*__retrain_without__value-object_hue-0.rival.json
% (PLACEHOLDER: w15 single-design-seeds; CLU n = 1)
\includegraphics[width=\linewidth]{figures/fig_deletion.pdf}

% SOURCE: w13-encoder-adapt/CROSS-ARM.json  (3 seeds; opponent mlp_180k)
\includegraphics[width=0.55\linewidth]{figures/fig_encoder_adapt.pdf}

% SOURCE: w14-read-verb/runs/scored{1,2,3}_seed{0,1,2}_test.json  (PLACEHOLDER: w16 maze-wm)
\includegraphics[width=\linewidth]{figures/fig_maze_readverb.pdf}

% SOURCE: w14-request-layer-fix/artefacts/measure_{before_0,after_0,after_1,after_2}.json  (PLACEHOLDER; appendix)
\includegraphics[width=\linewidth]{figures/fig_maze_requestlayer.pdf}
```

Compiled against `iclr2027/iclr2027_conference.sty` with pdflatex (TeX Live 2026): all six
include at `\linewidth`, 0 errors, 0 overfull boxes.
