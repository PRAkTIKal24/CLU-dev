# pw1-plot-maze — FINDINGS (PW-1 items 1, 16, figure half of 8; retirements from items 14, 15, 17)

Branch `pw1/plot-maze` (from `w22/figures3` cb5e0ac). Worktree `~/clu/worktrees/CHLU-pw1-plot-maze`. Nothing was installed into `w17-paper/figures/` or `paper-visuals/`, and neither draft was edited (per the brief). The build is in `build/a/` and `build/b/` of this directory.

## Item status

| item | status | what |
|---|---|---|
| 1.1 rows, Transformer-REL comparator | done | rows: words, full design · observation · words + observation (n-CLU) · words + observation (n-CLU-REL); the asymmetric row and the main-effects row are removed (`data/maze_conditions.json` `retired_rows`) |
| 1.2 every seed, count printed | done | 3 / 3 / 6 / 6 seeds drawn; the sub-label prints "N seeds (seeds 0–k)" |
| 1.3 no label column | done | head_label() is no longer drawn; it is only asserted to pick the same comparator for the 3-seed rows |
| 1.4 no overlay in (b)/(c); x = seed numbers; (c) REL only | done | tick labels are "0".."5", xlabel "seed"; (c) draws n-CLU-REL, Transformer-REL and the highest control only |
| 1.5 no ceiling in the figure | done (figure); listed (drafts) | no ceiling line or legend entry. Draft occurrences are listed with line numbers in `CEILING-ROWS.md` (19 per draft). The task paragraph is `TEXT-INSERTS.tex` (b) |
| 1.6 titles | done | the Head's titles verbatim, with "(a)/(b)/(c)" prefixes and CLU → n-CLU |
| 1.7 legends | done, with one deviation | (a): n-CLU, Transformer, Transformer-REL, "mean of 3 or 6 seeds". **LSTM is not listed** because the legend lists only the families drawn. With six seeds the best matched baseline of the words + observation row is the Transformer at $10\times$ training steps (six-seed mean 0.5490), not the LSTM (0.5351), so no row draws an LSTM. **"mean of 6 seeds" → "mean of 3 or 6 seeds"**, because two rows have 3 seeds. (b), (c): verbatim with the n-CLU tokens |
| 1.8 sub-labels | done | "best baseline: Transformer, $10\times$ training steps"; for the words-full row the comparator is the 8,000-step (declared) arm, so it reads "declared training steps"; the n-CLU-REL row reads "baseline: Transformer-REL, $10\times$ training steps" with seeds 0–5 |
| 1.9 glyph / no overlaps | done | the title has no glyph; the legend markers are the drawn markers. An automatic overlap-and-clipping assertion runs on every row label (`cmazet_panels.check_no_overlap`) |
| 1.10 caption | done | `CAPTIONS.tex` fig:maze-wm; banned-word grep empty |
| 16.1 no bottom paragraph; strips stay | done | `grid(key_line=False)`; the paragraph is in `TEXT-INSERTS.tex` (a) with the pointer sentence |
| 16.2 door successes by rule | done | see "Door search" below. The base n-CLU class is empty on seeds 0–2 (0 of 204), and seeds 3–5 cannot be searched. n-CLU-REL: 44 of 55 at seed 2, 0 on every other seed |
| 16.3 names | done | n-CLU / n-CLU-REL; Transformer (never Transf.); groups "matched baselines, 8,000 steps / 80,000 steps", "controls", "blind arms"; -REL baselines "MLP-REL / LSTM-REL / Transformer-REL" (group "baselines with relational columns"); control names in plain words ("landscape removed", "landscape shuffled", "shuffled across wells", "random atoms"); rule families "door then tint", "rule A", "rule B", "open door, tint", "free" |
| 16.4 gate, ambiguity, 2 builds | done | see "Checks" below |
| 8 (figure half) | done | no hard/easy shading, ticks, cut or column in any maze figure; `check_vocab` bans \bhard\b/\beasy\b in every rendered string |
| retire readverb / requestlayer (14) | **not done in build.sh (ownership)** | both are dispatched from `make_figures.py` `main()` lines 543–544, a file this spoke may not edit. My own build (`tools/build_maze_only.py`) does not build them. **The one-line fix for the owner:** delete those two calls |
| retire fig_maze_rel_tuples (15) | done | its dispatch in `make_figures_w22.main` is removed; the function is kept (not deleted) |
| retire fig_cmazet_deletion (17) | done | `status: "retired"` in `data/cmazet_panels.json`; the old files are not deleted |

## Decisions taken (pre-authorised; closest to an ICLR main-track figure)
1. **Door rows come from another model's run.** The base n-CLU has no door success, so the families door row and the door-then-tint success rows come from the n-CLU-REL run (seed 2). Every column of those rows is read from that run's own per-item file: its baselines, controls and blind arms. The REL runs retrained their baselines, and the verdicts differ from the base run's. The row label says so ("n-CLU-REL run, seed 2: all columns"), and a thin rule separates those rows from the rows above. The alternative was to leave Fig. 10 without a door row, which the Head's text rules out when such items exist.
2. **Per-arm figure (Fig. 11, base n-CLU):** the door class is empty, so the n-CLU success row stays the existing uniform draw. No substitution; the caption says why.
3. **REL per-arm figure (Fig. 12):** the n-CLU-REL "succeeds" row is a door success drawn by rule. The seed-0 uniform draw is still consumed, so every other row is byte-for-byte the same draw as before (recorded: `replaces_uniform_draw_i: 226`).
4. **"Door opening" vs "door-then-tint":** on these banks the two classes are the same set. Every held-out n-way row with a key consumed at a door is an `interaction` row (asserted per seed: `door_open_equals_door_then_tint: true`), because the n-way hold-out has no other door-opening family.
5. **Colour diversity is impossible** for the door draws: all 44 n-CLU-REL door successes have pocket colour 2 (E 13, S 14, W 17; it is wrong on all 11 northward rows of colour 1). The draw is stratified over (family, colour), and the per-direction draw gives direction diversity. The most-baselines-wrong draw is stratified over (family, colour) among 199 items.
6. **Draw seed 16** (numpy default_rng, one generator, slots in the fixed order of `cmazet_door_search.ALLOCATION`). Chosen before any draw was looked at; it is the item number.
7. **Comparator in (a)/(b):** the matched baseline with the highest mean over the row's seeds, taken over 3 families × 2 step limits. This is the same rule as the old head_label R*, and the build asserts they are equal on the 3-seed rows. (b) draws that same fixed arm per seed (Transformer, $10\times$ training steps). **It is not the per-seed best**: Tab. cmazet-sym's "best matched rival" is a per-seed max over the six arms, so the per-seed values differ, e.g. seed 0: 0.562 (Transformer 80k) vs 0.627 (LSTM 80k). The caption points to the table for every arm.
8. **Transformer-REL step limit:** 80,000 steps (six-seed mean 0.8886 vs 0.8817 at 8,000).
9. **Placeholders:** the live captions carry two \PLACEHOLDERs for w16 arms (families, door-tint) and one for deletion, which goes with item 17. My captions do not carry the first two, because their live wording holds banned words. A vocabulary-clean wording is given as a comment block at the end of `CAPTIONS.tex` for the installing spoke or the Head.

## Door search (item 16.2) — `DOOR-SEARCH.json` (sha256 in SHA256-build1.txt)
Rule: see `rule` in the JSON (verbatim module docstring). Matched baselines = rival: {mlp, lstm, transformer} at 8,000 and 80,000 steps of the row's own run.

| model | seed | n-way rows | door open: correct / rows | door then tint: correct / rows | n-CLU right, >=4 of 6 matched wrong | same, vs the 6 -REL baselines |
|---|---|---|---|---|---|---|
| n-CLU | 0 | 761 | 0 / 70 | 0 / 70 | 159 (rule_b 159) | n/a |
| n-CLU | 1 | 742 | 0 / 79 | 0 / 79 | 4 (free 4) | n/a |
| n-CLU | 2 | 686 | 0 / 55 | 0 / 55 | 36 (free 36) | n/a |
| n-CLU-REL | 0 | 761 | 0 / 70 | 0 / 70 | 104 (free 82, rule_b 22) | 0 |
| n-CLU-REL | 1 | 742 | 0 / 79 | 0 / 79 | 242 (open_door_tint 14, rule_b 228) | 247 |
| n-CLU-REL | 2 | 686 | 44 / 55 | 44 / 55 | 332 (interaction 44, open_door_tint 24, rule_b 264) | 191 |
| n-CLU-REL | 3 | 675 | 0 / 49 | 0 / 49 | 171 (open_door_tint 11, rule_a 4, rule_b 156) | 156 |
| n-CLU-REL | 4 | 723 | 0 / 72 | 0 / 72 | 253 (free 74, open_door_tint 21, rule_b 158) | 102 |
| n-CLU-REL | 5 | 660 | 0 / 40 | 0 / 40 | 78 (open_door_tint 6, rule_b 72) | 0 |
| n-CLU | 3, 4, 5 | NOT SEARCHABLE: no persisted objects, no per-item scores | | | | |

| slot | class | model | qualifying | drawn (seed, item i, family, pocket colour, action) | empty for |
|---|---|---|---|---|---|
| fig_cmazet_families: door row | door_then_tint | n-CLU-REL | 44 | s2 i467 interaction c2 S | n-CLU |
| fig_cmazet_families: most-baselines-wrong row | most_baselines_wrong | n-CLU | 199 | s0 i308 rule_b c1 W | - |
| fig_cmazet_rel_per_arm: n-CLU-REL success row | door_then_tint | n-CLU-REL | 44 | s2 i581 interaction c2 E | - |
| fig_cmazet_door_then_tint: success rows | door_then_tint | n-CLU-REL | 44 | s2 i214 interaction c2 E; s2 i332 interaction c2 S; s2 i348 interaction c2 W | n-CLU |

The n-CLU (base) door class is **empty** at seeds 0–2: 0 of 204. Seeds 3–5 of the base n-CLU are **not searchable**. `~/clu/stage/w19-persist/` holds BOTH_seed0..2 only, and `data/cmazet_items.json` covers seeds 0–2, so no per-item scores exist for seeds 3–5 and nothing was re-scored.

## Numbers printed in the new figures and captions (verified against the source)
Per-row values of fig_maze_conditions (PROVENANCE-maze.json → figures.fig_maze_conditions.rows / six_seed). Every value is read from the artefact, and the build asserts it float-equal:

| row | seeds | n-CLU per seed | comparator | comparator per seed |
|---|---|---|---|---|
| words, full design (TEST) | 0–2 | 0.8003 / 0.5660 / 0.8076 | rival: transformer (8k, declared) | 0.8252 / 0.5674 / 0.8338 |
| observation (VAL) | 0–2 | 0.2407 / 0.3733 / 0.4399 | rival@10x-steps: transformer | 0.4641 / 0.4504 / 0.4554 |
| words + observation, n-CLU (TEST) | 0–5 | 0.8003 / 0.5660 / 0.8076 / 0.6741 / 0.5325 / 0.6818 | rival@10x-steps: transformer (six-seed mean 0.5490) | 0.5624 / 0.5566 / 0.6618 / 0.4074 / 0.4302 / 0.6758 |
| words + observation, n-CLU-REL (TEST) | 0–5 | 0.9080 / 0.8935 / 0.9840 / 0.9274 / 0.9004 / 0.9394 | diagnostic+rel@80k: transformer +rel (six-seed mean 0.8886) | 0.9080 / 0.8693 / 0.8090 / 0.9052 / 0.9004 / 0.9394 |

Printed as observed, not interpreted: the n-CLU's seed 0–2 values on the full word design (BARS-WORDS-FULL-test.json) are float-identical to its seed 0–2 values on words + observation (BARS-BOTH-test.json / the w19 run JSONs): 0.8002628120893561 / 0.5660377358490566 / 0.8075801749271136 in both. The Transformer-REL ties the n-CLU-REL on seeds 0, 4 and 5, and the highest control reaches 0.9004 on seed 4 (= n-CLU-REL; random atoms). These are visible in (c) as coincident markers, with no text over them (item 1.4).

## Checks (all run, all passing)
- **Two builds byte-identical**: `SHA256-build1.txt` == `SHA256-build2.txt` (12 files: 5 figures × pdf/png, DOOR-SEARCH.json, PROVENANCE-maze.json).
- **Score gate** (pure read, `tools/build_maze_only.py gate()`): every arm's n-way score recomputed from the per-item files (correct / n) is float-equal to the run JSON's ROWS.nway. That is 48 arms for the base n-CLU (seeds 0–2) and 132 for n-CLU-REL (seeds 0–5); every run JSON's sha256 equals the one the per-item file recorded. Flag: the `.REPRO.json` files do not carry the items files' own sha256 (`repro_mentions_this_sha: false`); the items sha256 is recorded in PROVENANCE instead.
- **Staged-object hash gate** (`cmazet_verify.py`, hash-only, nothing loaded): w19-persist 132/132, rel-persist clu-improve seeds 0–2 96/96, rel-newseeds seeds 3–5 96/96. ALL HASHES MATCH (`gate/HASH-VERIFY-*.json`).
- **Ambiguity** (`cmazet_ambiguity.py`): same-picture opposite-mark pairs with the strip (pixels+strip) = **0** for all arms and for drawn arms, both items files. The output is byte-identical to the committed `data/CMAZET-AMBIGUITY.json` (`gate/CMAZET-AMBIGUITY.json`).
- **Maze-six asserts** (maze_six.read): every per-seed arm is float-equal to REL-NEWSEEDS.json. The six-seed n-CLU row's seeds 0–2 are asserted equal to BARS-BOTH-test.json for the CLU and all six matched arms.
- **Vocabulary**: `cmazet_panels.check_vocab` asserts on every rendered string of all 5 figures (rival, store, gate, artefact, A1, A10, ceiling, hard, easy, exact, corner, improved, config, budget, bare "CLU", "Transf.", "+rel", internal control names). The banned-word grep of CAPTIONS.tex / TEXT-INSERTS.tex outside comments is empty.
- **Overlap / clipping**: asserted on every header, group label and row label of the panel figures, and on panel (a)'s row labels. I also looked at every PNG.
- **Full pipeline** (`build.sh` into scratch dests only): exit 0, "REBUILD: BYTE-IDENTICAL". Its fig_maze_conditions / fig_cmazet_{families,per_arm,rel_per_arm,door_then_tint} are cmp-equal to `build/a/`. fig_maze_rel_tuples and fig_cmazet_deletion are no longer produced; fig_maze_readverb and fig_maze_requestlayer still are (see above).

## Hashes (build/a; identical in build/b)
baeceb0f1212c40876066e8f005915dc793ed091eb0d67a2fc30a7b8dc44ef41  DOOR-SEARCH.json
ea36760fe82cfc2d38c0d080a25cb1c24979dcc08fc780de4dc66bbeeba193b8  fig_cmazet_door_then_tint.pdf
8900a6dfff9bd4f40853bb6266fbe3a230d3ee025241180f72565528ca3b27c6  fig_cmazet_door_then_tint.png
b774ac729782480d5c279a62910f3259dd4b87ece63aab7bf9bc174f891e17ae  fig_cmazet_families.pdf
bc029153e19370f9e1e2eaf98e8b85ef32996b391e571ab2506f29f47755d53f  fig_cmazet_families.png
744e643df043307e89844f011d2195dd1ac55b4daaad05ce0ba6216a35c278a6  fig_cmazet_per_arm.pdf
000ed77447ccecbfc3b49ee4689969da04e6ee2bc8c63896f8fb69853462d45b  fig_cmazet_per_arm.png
22454719c5f85e3ad0d4d443d0402019c234c5716aedd980ba66212e9ad31d23  fig_cmazet_rel_per_arm.pdf
6e33878c8313949660f7077c5a89e0bb1a1eb38a25037ccdc1d27bccb3a20c5a  fig_cmazet_rel_per_arm.png
3896e8128efb912bcdc9ec8f22909d7de67d3930671fe22f1a0f6a10eeee47e6  fig_maze_conditions.pdf
f4815379a04d9adea9e54a63f04f3f6a9f6c3d3f62a862a4cb2743f6cf2bab21  fig_maze_conditions.png
56b43fe99cd8d4b5ae2e6b728e01e70b2f204a9ec0867f4dddb27178bb6b0bba  PROVENANCE-maze.json

## Files changed on the branch (code only)
- `w21_figures/make_figures_w22.py`: fig_maze_conditions rewritten; the fig_maze_rel_tuples dispatch removed; docstring. Now-unused module constants `RIVAL_C`, `HARD_BG` left in place (minimal diff; not shared helpers I may edit freely).
- `w21_figures/cmazet_panels.py`: no bottom paragraph; family names; drawn rows / override rows; vocabulary, overlap and clipping checks; separator rule.
- `w21_figures/cmazet_door_search.py` (new), `w21_figures/tools/build_maze_only.py` (new).
- `w21_figures/data/cmazet_panels.json`, `w21_figures/data/maze_conditions.json`.
- Not touched: `make_figures.py`, `build.sh`, `make_figures_w21.py`, shared palettes / label dictionaries. `make_figures_w21.fig_maze_conditions` (old, not built by build.sh) is unchanged.

## What did NOT run
- No model was loaded and nothing was scored, fitted or re-extracted: zero runs, zero GPU. The base n-CLU seeds 3–5 per-item search is therefore **absent**, not zero; producing it would need persisted objects that do not exist, i.e. a new run.
- No install into `w17-paper/figures/` or `paper-visuals/`, no draft edit, no README update, no PDF compile of the paper (all out of scope per the brief; a later spoke installs). The `.tex` lock guard was therefore not needed.
- `fig_maze_readverb` / `fig_maze_requestlayer` are not retired from `build.sh`'s output (the file that dispatches them is not mine).
- fig_task_cmazet and every non-maze figure were built only as part of the scratch full-pipeline check; not inspected, not delivered.
- `TEXT-INSERTS.tex` and `CAPTIONS.tex` were not compiled in the paper (the later spoke builds them). Their LaTeX uses only commands already in the draft (\figslot, \ref, \paragraph, math).

## Commit and bundle
Branch `pw1/plot-maze` at 2b1791e (parent cb5e0ac). Bundle `pw1-plot-maze.bundle` sha256 058a8f3fc68ebe9d9a82728d01d6db8c4c8fbe08855f138d3de03373498b3b90; `git bundle verify`: complete history, OK. The code hashes in PROVENANCE-maze.json (code_sha256) match the committed files: no edit was made after the build.
