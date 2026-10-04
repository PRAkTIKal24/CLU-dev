# FINDINGS — pw1-visuals-tooling (PW-1 item 4: `paper-visuals/` rebuilds itself in place)

Session `pw1-visuals-tooling` · branch `pw1/visuals-tooling` (from `pw1/draft-figures` a01bea7) · final commit `f8ef2bb` · batmac · start 2026-09-25T12:11:22Z.
No draft and no PDF written. No figure byte changed. Zero runs, zero fits, zero GPU. The figure builds read artefacts only.

## Status
| job | status | note |
|---|---|---|
| Fix the repo-root derivation (`cmazet_panels.py`, `build.sh`, and anything else that assumes the folder's parent is a CHLU repo) | **done** | new `figroot.py` resolves the layout (repo / flat folder) and the engine root (`$CHLU_FIG_CODE` > repo > the folder's `engine/`). Fixed: `cmazet_panels._engine`, `build.sh` (`cd $REPO && python -m w19_task_figures…`, `cp` onto itself in place), `verify_glyph.py` (sibling `w21_figures/`, package import, data path), `tools/merge_task_prov.py` (REPO-relative source hashes), `cmazet_extract_rel.py` (not in the build; now `CHLU_FIG_CODE` + a clear exit) |
| `bash build.sh` in place in `~/clu/outputs/paper-visuals/`, byte-identical to the installed files | **done** | exit 0; a = b; **20 / 20 figure files + GLYPH-IDENTITY.json equal the installed files** in both folders; `PROVENANCE.json` differs by exactly three Hub-mandated / honest edits (below) |
| Works from a fresh copy (`/tmp/…`) | **done** | copy of the INSTALLED folder under `env -i` (no CHLU_* vars, no PYTHONPATH, C locale): exit 0, manifests **byte-equal** to the in-place build's. Copy without `engine/`: exit 1 before drawing, with the message naming `engine/` and `CHLU_FIG_CODE` and pointing to the README |
| Mirrored figures: one documented rebuild line each (dataset, frozen-decoder env) | **done** | `fig_task_shapes`: line tested, reproduces the installed bytes (two builds + the literal line). `rendered/rc_pw1_*`: line documented, inputs checked present, **not re-rendered** (What did NOT run) |
| README states the one command at the top | **done** | both folders; `make_index.py` now imports the same block (`pw1_visuals_tooling/readme_block.py`), and a regenerated README is byte-equal to the patched one |
| Hub note: stop producing `fig_headline_steps` (dispatch removed, function kept); 13 placed stems byte-unchanged | **done** | 26 / 26 placed files unchanged in each folder; the build now makes 10 stems (was 11) |
| Test in a copy first → commit → install scripts into both folders → in-place build once to scratch | **done** | order kept; build.sh was re-installed once after a locale fix (below), and the in-place run was repeated with the final build.sh (`inplace-final-122755/`); the first run (`inplace-122459/`) is kept |

## What was wrong (before), observed
`before-121538/`: a scratch copy of the installed `paper-visuals/`, old `build.sh`, scratch destinations: **exit 1**, `ModuleNotFoundError: No module named 'cmazet_door_search'` (imported by `cmazet_panels.build`, never installed in the folder). Past that point, reading the code, it would have failed in three more places:
- `cd $REPO && python -m w19_task_figures.make_task_figures`, with REPO = `~/clu/outputs`;
- `cmazet_panels._engine`'s assert that chlu lives under the folder's parent;
- `verify_glyph`'s `../w21_figures/data`.

A further hazard: the venv has an editable `chlu` pointing to `~/clu/workspace` at 148f3b7. Any import that is not pinned would silently use that older engine. `figroot.check_loaded()` now asserts that every engine module loads from the engine root.

## Design (choices taken, logged)
1. **Engine snapshot inside the folder** (`engine/`, 61 files, 1.4 MB). This is the import closure of `chlu.exp_w15_cmaze_t`, `w16_cmaze_t`, `w8_envscout.maze_colour` and `w9_harness.dynamics`, byte-copied by `tools/vendor_engine.py`. `engine/ENGINE.json` records the source commit and the sha256 of each file. The copy is idempotent: an unchanged snapshot is not rewritten, and a changed one is moved aside, never deleted. I chose this over a hard-coded checkout path because worktrees are transient, and the item's title is "self-contained". The 61 files are byte-identical at f571b2b and HEAD (`git diff --stat f571b2b HEAD` over them is empty).
2. **The external inputs that remain are named in the README:**
   - the evidence tree `$CHLU_EVIDENCE` (default `~/clu/evidence/outputs`, the same string PROVENANCE records);
   - the project venv's `python` (numpy, matplotlib 3.10.8, jax/equinox, because `chlu/__init__` imports them).

   `--assets` is passed but read only by `fig_task_shapes`, which `build.sh` never draws.
3. **Default destinations are unchanged**: `w17-paper/figures` and `paper-visuals`. When a destination is the folder itself, code is not copied, so nothing is copied onto itself. `PYTHONDONTWRITEBYTECODE=1`: the in-place build left the `paper-visuals/` tree byte-unchanged, `__pycache__` included (pre/post tree hashes equal).
4. **The installed `PROVENANCE.json` is NOT overwritten.** The brief says "no installed file is overwritten", so it stays the f571b2b build's. The Head's first real `bash build.sh` will install the new one. That file equals the installed one with exactly three edits (checked by applying them and comparing bytes: **True**):
   - `figures.fig_headline_steps` is removed;
   - `"fig_headline_steps"` is removed from `figures_redraw.stems_redrawn_or_added`;
   - `figures.fig_task_cmazet.source_sha256["w21_figures/cmazet_panels.py"]` changes from `59fe2a674f034609…`, the sha of the old `cmazet_panels.py`, to `125b40dd567acc44…`, the sha of the fixed file now installed.

   Until then, the installed PROVENANCE records the old `cmazet_panels.py` hash.
5. **Manifest order pinned to the C locale.** The first fresh-copy test under `env -i` wrote the same hashes in a different line order, because `sort` depends on the locale. `build.sh` now uses `LC_ALL=C sort`. Consequence: the next install writes `SHA256-w22-build(1, 2).txt` with `GLYPH-IDENTITY.json` and `PROVENANCE.json` first. These manifests differ from the installed ones anyway, since they drop the two headline lines and change the PROVENANCE line.
6. **`.tex` lock guard** printed `ABORT: pw1-draft-wording live` before the install. That guard covers draft writes. This spoke writes no draft, and the brief names draft-wording as the parallel `.tex`-only spoke, so I went ahead with the scripts-only install and logged it.
7. `make_task_figures.py` is **unchanged**, so its source hash in PROVENANCE is unchanged. It runs as a script, with the engine on `PYTHONPATH`, in both layouts.

## Acceptance

### In-place build (`cd ~/clu/outputs/paper-visuals && bash build.sh <scratch1> <scratch2>`, final build.sh; `inplace-final-122755/`)
exit 0 · `engine OK: …/paper-visuals/engine (folder layout), 61 modules from it` · `REBUILD: BYTE-IDENTICAL` · `INSTALLED BYTE-IDENTICAL` ×2 (scratch) · paper-visuals tree unchanged.

build a == build b (manifests): True | files: 22
| file | build a | paper-visuals (installed) | figures (installed) | equal |
|---|---|---|---|---|
| GLYPH-IDENTITY.json | 671f706c4b990022 | 671f706c4b990022 | 671f706c4b990022 | YES |
| PROVENANCE.json | ec5a61223d5e6681 | b2e318176ef8880b | b2e318176ef8880b | **NO** |
| fig_cmazet_door_then_tint.pdf | ea36760fe82cfc2d | ea36760fe82cfc2d | ea36760fe82cfc2d | YES |
| fig_cmazet_door_then_tint.png | 8900a6dfff9bd4f4 | 8900a6dfff9bd4f4 | 8900a6dfff9bd4f4 | YES |
| fig_cmazet_families.pdf | b774ac729782480d | b774ac729782480d | b774ac729782480d | YES |
| fig_cmazet_families.png | bc029153e19370f9 | bc029153e19370f9 | bc029153e19370f9 | YES |
| fig_cmazet_per_arm.pdf | 744e643df043307e | 744e643df043307e | 744e643df043307e | YES |
| fig_cmazet_per_arm.png | 000ed77447ccecbf | 000ed77447ccecbf | 000ed77447ccecbf | YES |
| fig_cmazet_rel_per_arm.pdf | 22454719c5f85e3a | 22454719c5f85e3a | 22454719c5f85e3a | YES |
| fig_cmazet_rel_per_arm.png | 6e33878c83139496 | 6e33878c83139496 | 6e33878c83139496 | YES |
| fig_crossover.pdf | 0dfda2510acbce7f | 0dfda2510acbce7f | 0dfda2510acbce7f | YES |
| fig_crossover.png | 22262422cea96dd8 | 22262422cea96dd8 | 22262422cea96dd8 | YES |
| fig_deletion.pdf | 8faeaf9fa28f05d8 | 8faeaf9fa28f05d8 | 8faeaf9fa28f05d8 | YES |
| fig_deletion.png | b444cfe683d2669a | b444cfe683d2669a | b444cfe683d2669a | YES |
| fig_encoder_adapt.pdf | 3c0a924da28bf664 | 3c0a924da28bf664 | 3c0a924da28bf664 | YES |
| fig_encoder_adapt.png | fe565909d28da32a | fe565909d28da32a | fe565909d28da32a | YES |
| fig_ladder.pdf | 949d5b6d1ff2fac3 | 949d5b6d1ff2fac3 | 949d5b6d1ff2fac3 | YES |
| fig_ladder.png | a5d05a35d53e5272 | a5d05a35d53e5272 | a5d05a35d53e5272 | YES |
| fig_maze_conditions.pdf | 3896e8128efb912b | 3896e8128efb912b | 3896e8128efb912b | YES |
| fig_maze_conditions.png | f4815379a04d9ade | f4815379a04d9ade | f4815379a04d9ade | YES |
| fig_task_cmazet.pdf | f701f98fa9a9f0ab | f701f98fa9a9f0ab | f701f98fa9a9f0ab | YES |
| fig_task_cmazet.png | 10c9aa59c18611fd | 10c9aa59c18611fd | 10c9aa59c18611fd | YES |

equal: 21 / 22. The one difference is `PROVENANCE.json`, and it is exactly the three edits of choice 4. Build a = build b (manifests identical). Full 64-hex manifest: `inplace-final-122755/dest2/SHA256-w22-build1.txt`.

### The same command in other places
| run | exit | manifest vs in-place final |
|---|---|---|
| repo layout, `bash w21_figures/build.sh` from the worktree (`repo-build-121752/`) | 0 | same 22 hashes; PROVENANCE `ec5a61223d5e6681` |
| scratch copy of the folder, scripts installed, `env_bm.sh` sourced in the folder (`copytest-121901/`) | 0 | same 22 hashes |
| in place in `~/clu/outputs/w17-paper/figures/` (`inplace-w17figures-122559/`) | 0 | same 22 hashes; w17 tree unchanged |
| fresh `/tmp` copy of the installed folder, `env -i` (`/tmp/pw1-visuals-fresh3-122755`, output copied to `fresh3-122755/`) | 0 | **manifest files byte-equal** |
| fresh `/tmp` copy with `engine/` moved away | **1** | prints: `figroot: no maze engine found (tried engine/=…). The CMAZE-T figures need the engine code: a folder copy carries it in engine/ (README.md, 'Rebuild'); or set CHLU_FIG_CODE to a CHLU checkout at f571b2b or later.` |
| same copy with `CHLU_FIG_CODE=<worktree>` | 0 | outputs identical to the fresh build |

### Install (scripts only) into both folders; trees before → after (`snap-before/TREE-*.sha256`, `TREE-final-*.sha256`)
| folder | files before → after | changed | removed | new | non-code files unchanged | 13 placed stems unchanged |
|---|---|---|---|---|---|---|
| paper-visuals/ | 75 → 141 | 6: README.md, build.sh, cmazet_extract_rel.py, cmazet_panels.py, make_figures_w22.py, verify_glyph.py | 0 | cmazet_door_search.py, figroot.py, tools/merge_task_prov.py, tools/vendor_engine.py + 62 under engine/ | 63 / 63 | 26 / 26 files |
| w17-paper/figures/ | 108 → 174 | same 6 | 0 | same | 92 / 92 | 26 / 26 files |
Every installed script is `cmp`-equal to the branch copy. Old copies of the six changed files are in `snap-before/<folder>/`.

Installed code hashes (first 16 hex, identical in both folders): build.sh 95bb559a91e60f1e · figroot.py 93e993dd86a80c4e · cmazet_panels.py 125b40dd567acc44 · make_figures_w22.py c7ab7f26b06be17c · verify_glyph.py 15f5b5605066c23d · cmazet_extract_rel.py d6f76532a51a9ee3 · cmazet_door_search.py b999254d38a53195 · tools/merge_task_prov.py db8006a4a70182cb · tools/vendor_engine.py fbabdd2d71470eaf · engine/ENGINE.json 2b3dd606f540cb37. README.md: paper-visuals c34ace2edfcb34a8, w17-paper/figures ce818d874ab07736.

### Mirrored figures
| stem | rebuild line | tested |
|---|---|---|
| `fig_task_shapes` | `cd ~/clu/outputs/paper-visuals && PYTHONPATH=.:engine/scripts:engine python make_task_figures.py --only all --assets ~/clu/outputs/w19-task-figures/assets --out /tmp/task-shapes && cp /tmp/task-shapes/fig_task_shapes.* .` | **yes**: pdf `c83b24d551ec128e`, png `78a8d7ed2995dd1b` = installed, two builds plus the literal line (run in the scratch copy). Assets: `shapes_images.npz` 9f9d71bc…, `task_assets.npz` bc6df90e…, `ASSETS.json` 8653edb5… (= the w19 record). Upstream stages (dataset `~/clu/data/gen/shapes3d/3dshapes.h5` in `h5venv`; frozen decoder `decoder_v1_s0.pkl` 01ef91f7… in the project venv) are documented, not re-run |
| `rendered/rc_pw1_shapes_{per_model,baseline_wins}` (+ REQUESTS-pw1.json, PROVENANCE-render.json) | `cd <CHLU checkout at a01bea7 or later> && bash pw1_render_shapes/build_twice.sh`, then copy `build/a` into `rendered/` | **no** (inference through the frozen decoder and the classifier). Every input path in `run_env.sh`/`common.py` checked present: w20 manifest, n-CLU `arm.pkl`, `decoder_v1_s0.pkl`, `winner_hashes.json`, classifier dir, split dir, `SHAPES-MATCHED.json`, 3dshapes.h5 |

## Gaps found (printed, not fixed: outside this item)
- `w19_task_figures/extract_assets.py` hard-codes `~/clu/outputs/w17-paper/figures/rendered/REQUESTS.json`. pw1-draft-figures moved that file to `rendered/retired-pw1/REQUESTS.json` (sha256 cf0acb8c…). Re-running Stage B would fail on that path. The README says so. The built `task_assets.npz` is unaffected.
- `pw1_render_shapes/build_twice.sh` sources `run_env.sh`, which forces `PW1_RS_OUT=~/clu/outputs/pw1-render-shapes`. A re-render writes there (earlier builds are moved aside) and cannot be redirected without editing it.
- `cmazet_extract.py` (w19 gate) is repo-only and not installed in the folders. It is unchanged.

## What did NOT run
- No re-render of `rendered/rc_pw1_*`. It needs model inference (frozen decoder, judge classifier, w20 objects) and falls outside a tooling item. Only the presence of its inputs was checked.
- No re-run of the w19 asset stages (`extract_images.py`, `extract_assets.py`). `fig_task_shapes` was rebuilt from the frozen assets only.
- The extraction scripts (`cmazet_extract_rel.py`, `cmazet_ambiguity.py`, `cmazet_door_search.py` standalone) were not run. Their outputs (`data/*.json`) are inputs to the build and are unchanged.
- No draft, PDF, `FIGURES.md` or `PROVENANCE.json` was written in either folder (see choice 4). `make_index.py` was run only on a scratch copy (with the pw1-draft-figures build-final aux), never on the live folders.
- No compile. No draft changed.

## Diff of script changes (code only; `git diff a01bea7 f8ef2bb`; also `SCRIPT-DIFF.patch`)
<details><summary>535 lines</summary>

```diff
diff --git a/pw1_draft_figures/make_index.py b/pw1_draft_figures/make_index.py
index f745dbf..70234a6 100644
--- a/pw1_draft_figures/make_index.py
+++ b/pw1_draft_figures/make_index.py
@@ -19,6 +19,10 @@ import re
 import sys
 
 PV, PAPER, AUX, OUT, COMMIT = sys.argv[1:6]
+# PW-1 item 4: the README's Rebuild section and the build-code bullet live in one place
+sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
+                                "pw1_visuals_tooling"))
+import readme_block  # noqa: E402
 DRAFTS = ["DRAFT-v2.tex", "DRAFT-v2-style.tex"]
 NAME = "paper-visuals" if PV.rstrip("/").endswith("paper-visuals") else "w17-paper-figures"
 W19TASK = "/Users/batmac/clu/outputs/w19-task-figures/proof"
@@ -48,7 +52,7 @@ DESCRIPTIONS = {
     "fig_cmazet_rel_per_arm": "Maze task with relational coding: as fig_cmazet_per_arm for the n-CLU-REL, the matched baselines, controls and blind arms of the same runs, and MLP-REL, LSTM-REL and Transformer-REL at 8,000 and 80,000 steps; the n-CLU-REL success row is a door success at seed 2.",
     "fig_cmazet_door_then_tint": "Maze task, door-then-tint moves (a door opens with the key and the move enters a tint of the same colour): rows 1-3 at seed 0 (north, south, west), where the n-CLU is wrong; rows 4-6 door successes from the n-CLU-REL run at seed 2 (east, south, west), below a rule; every model's next state with tick or cross and fact strip.",
     # retired (in retired-pw1/)
-    "fig_headline_steps": "Retired (no slot since w22 Ruling 2; still produced by build.sh): distinct valid novel tuples per held-out combination against the baselines' optimiser steps (log), 3 seeds.",
+    "fig_headline_steps": readme_block.RETIRED_HEADLINE,
     "fig_maze_readverb": "Retired (PW-1 item 14): route-task read-verb figure.",
     "fig_maze_requestlayer": "Retired (PW-1 item 14): request-layer audit figure.",
     "fig_maze_rel_tuples": "Retired (PW-1 item 15): per-held-out-tuple relational heat maps.",
@@ -221,6 +225,7 @@ def md(t):
 
 def write_readme(rows, ret_rows, c):
     L = ["# %s — figure index\n" % ("paper-visuals" if NAME == "paper-visuals" else "w17-paper/figures")]
+    L.append(readme_block.REBUILD_MD)
     L.append("Every figure file the two ICLR drafts (`w17-paper/DRAFT-v2.tex`, `DRAFT-v2-style.tex`) include, one line per file. "
              "\"Where used\" comes from each draft's `\\figslot{stem}{caption}{label}` calls and its `figure` environments with "
              "`\\includegraphics` (the rendered Shapes3D figures); figure numbers, section numbers and pages are read from the compiled "
@@ -244,14 +249,6 @@ def write_readme(rows, ret_rows, c):
                  ", ".join(c["files_here_unreferenced"]) or "none", c["retired_files"]))
     L.append("- **PNG/PDF parity:** %d stems, %s.\n" % (c["stems_here"], "every one has both" if not c["png_pdf_parity_failures"]
                                                         else "MISSING A FORMAT: " + ", ".join(c["png_pdf_parity_failures"])))
-    L.append("## Rebuild\n")
-    L.append("- Build figures: `cd <a CHLU worktree at %s or later>; source ~/clu/env_bm.sh; bash w21_figures/build.sh <dest1> <dest2>` "
-             "builds every build figure twice, checks byte-identity and installs into both destinations. The build also produces "
-             "`fig_headline_steps`, which has no slot and is moved to `retired-pw1/` after an install. "
-             "(Running `build.sh` in place in this folder is PW-1 item 4, not yet done.)" % c["build_commit"][:7])
-    L.append("- `fig_task_shapes`: mirrored from w19 task-figures (needs the Shapes3D dataset file); not rebuilt here.")
-    L.append("- `rendered/rc_pw1_*`: mirrored from pw1 render-shapes (`pw1_render_shapes/build_twice.sh`; needs the frozen decoder, "
-             "the judge classifier and the w20 objects). Their two-build manifest is `rendered/SHA256-pw1-render-build{1,2}.txt`.\n")
     L.append("## Figures (one line per file)\n")
     L.append("| file | what it shows | where used | data source | build hash (SHA-256) |")
     L.append("|---|---|---|---|---|")
@@ -267,8 +264,7 @@ def write_readme(rows, ret_rows, c):
     L.append("- `PROVENANCE.json`: per build figure, every source read with its SHA-256 (written by the build). "
              "`GLYPH-IDENTITY.json`: the task-figure / result-panel glyph identity check. `SHA256-w22-build{1,2}.txt`: the build's two-build manifest "
              "(the file name is the build script's).")
-    L.append("- `build.sh`, `make_figures*.py`, `maze_six.py`, `cmazet_*.py`, `make_task_figures.py`, `verify_glyph.py`, `validate_palette.py`, "
-             "`data/*.json`: the build code and its data, copied by the build.")
+    L.append(readme_block.OTHER_CODE_MD)
     L.append("- `rendered/PROVENANCE-render.json`, `rendered/REQUESTS-pw1.json`: the rendered figures' provenance, drawn requests, the judge's "
              "six read-backs per panel, and the counts quoted in the captions.")
     L.append("- `FIGURES.md`: longer notes per figure; its PW-1 section is on top.")
diff --git a/w19_task_figures/verify_glyph.py b/w19_task_figures/verify_glyph.py
index d9fdc01..6169f20 100644
--- a/w19_task_figures/verify_glyph.py
+++ b/w19_task_figures/verify_glyph.py
@@ -18,10 +18,20 @@ import sys
 import numpy as np
 
 HERE = os.path.dirname(os.path.abspath(__file__))
-sys.path.insert(0, os.path.join(os.path.dirname(HERE), "w21_figures"))
+# PW-1 item 4: repo layout (w19_task_figures/ next to w21_figures/) or a flat figure folder
+FIGS = os.path.join(os.path.dirname(HERE), "w21_figures")
+if not os.path.isfile(os.path.join(FIGS, "figroot.py")):
+    FIGS = HERE
+sys.path.insert(0, FIGS)
+import figroot  # noqa: E402
+figroot.add_code_to_path()
 import cmazet_panels as CP  # noqa: E402
-from w19_task_figures import make_task_figures as TF  # noqa: E402
+if FIGS == HERE:
+    import make_task_figures as TF  # noqa: E402
+else:
+    from w19_task_figures import make_task_figures as TF  # noqa: E402
 from w9_harness.dynamics import State  # noqa: E402
+figroot.check_loaded()
 
 C = CP.CELL
 yy, xx = np.mgrid[0:C, 0:C]
@@ -33,8 +43,7 @@ def cell(img, r, c, r0=0, c0=0):
 
 
 def panel_agent_block(pocket_want):
-    d = json.load(open(os.path.join(os.path.dirname(HERE), "w21_figures", "data",
-                                    "cmazet_items.json")))
+    d = json.load(open(os.path.join(FIGS, "data", "cmazet_items.json")))
     order = sorted(d["seeds"], key=lambda s: (s["seed"] != 2, s["seed"]))
     for srec in order:
         mz = CP.Maze(srec)
diff --git a/w21_figures/build.sh b/w21_figures/build.sh
index a15cfc2..016fc5e 100755
--- a/w21_figures/build.sh
+++ b/w21_figures/build.sh
@@ -1,58 +1,85 @@
 #!/usr/bin/env bash
-# Build the figures twice into scratch dirs, prove byte-identity, then install into the
-# shared draft's figures/ dir (never touching figures/rendered/, which another spoke owns).
+# Build the figures twice into scratch dirs, prove byte-identity, then install into two
+# destinations (the draft's figures/ dir and paper-visuals/; never figures/rendered/).
 #
-# w21 figures-redraw extends this build (it does not replace it):
-#   1. make_figures.py      -- the w17 figures, now with NO in-image banner (tag_placeholder is a no-op)
-#   2. make_figures_w21.py  -- redraws fig_crossover / fig_ladder / fig_deletion over step 1's files,
-#                              adds fig_maze_conditions, and (Part C) fig_cmazet_* when the verified
-#                              per-item extraction data/cmazet_items.json is present
-#   3. PROVENANCE.json      -- step 1's entries, overridden per stem by step 2's (merge, never drop)
-# w22 figures-redraw pass 2: step 2 is make_figures_w22.py (it imports make_figures_w21.py as a
-# library and adds fig_headline_steps / fig_maze_rel_tuples; maze_six.py reads the six-seed CMAZE-T
-# artefacts). A second destination (paper-visuals) receives the same bytes.
-# Rows / rival sets / panel sets are DATA: data/maze_conditions.json, data/ladder_rival_sets.json,
+# PW-1 item 4: this script runs in either layout (figroot.py):
+#   repo   -- <CHLU repo>/w21_figures/build.sh; the maze engine is the repo;
+#   folder -- IN PLACE in outputs/paper-visuals/ (or w17-paper/figures/, or any copy of them):
+#             every script sits next to this file and the maze engine is the folder's engine/.
+#   usage:  bash build.sh [dest1] [dest2]
+#           defaults: dest1 = ~/clu/outputs/w17-paper/figures, dest2 = ~/clu/outputs/paper-visuals
+#   External inputs (read-only): the evidence tree $CHLU_EVIDENCE (default ~/clu/evidence/outputs)
+#   and the project venv's python (numpy, matplotlib 3.10.8, jax/equinox for the engine's imports).
+#
+# Steps, per build (a and b):
+#   1. make_figures.py      -- the w17 figures (fig_encoder_adapt; the others are overwritten by 2)
+#   2. make_figures_w22.py  -- fig_crossover, fig_ladder, fig_maze_conditions, fig_deletion and the
+#                              fig_cmazet_* panels (data/cmazet_items*.json, data/cmazet_panels.json)
+#   3. make_task_figures.py --only cmazet -- fig_task_cmazet (fig_task_shapes is never built here),
+#      then verify_glyph.py (task-figure agent == result-panel agent) and the provenance merge.
+# fig_headline_steps is no longer built (no slot; retired-pw1/, PW-1 item 4).
+# Rows / baseline sets / panel sets are DATA: data/maze_conditions.json, data/ladder_rival_sets.json,
 # data/cmazet_panels.json. Adding one is an edit there plus a rebuild, not a code change.
-# w22 pass 3: + fig_task_cmazet (w19_task_figures, --only cmazet; fig_task_shapes is never rebuilt
-# here) and its glyph-identity gate; fig_cmazet_rel_per_arm from data/cmazet_items_rel.json.
 set -euo pipefail
 HERE="$(cd "$(dirname "$0")" && pwd)"
 DEST="${1:-/Users/batmac/clu/outputs/w17-paper/figures}"
 DEST2="${2:-/Users/batmac/clu/outputs/paper-visuals}"
-TMP="${TMPDIR:-/tmp}/w21-figures-build.$$"
+PY="${PYTHON:-python}"
+LAYOUT="$("$PY" "$HERE/figroot.py" layout)"
+CODE="$("$PY" "$HERE/figroot.py" code)"
+if [ "$LAYOUT" = repo ]; then
+  REPO="$(dirname "$HERE")"
+  TASKFIG="$REPO/w19_task_figures/make_task_figures.py"
+  VERIFY="$REPO/w19_task_figures/verify_glyph.py"
+else
+  TASKFIG="$HERE/make_task_figures.py"
+  VERIFY="$HERE/verify_glyph.py"
+fi
+MERGE="$HERE/tools/merge_task_prov.py"
+# the engine root first, so no other checkout on sys.path (e.g. the venv's editable chlu) is used
+export PYTHONPATH="$HERE:$CODE/scripts:$CODE"
+export PYTHONDONTWRITEBYTECODE=1
+"$PY" "$HERE/figroot.py" check
+# the assets dir is read only by fig_task_shapes, which this build never draws (--only cmazet)
+ASSETS="${W19_TASK_ASSETS:-/Users/batmac/clu/outputs/w19-task-figures/assets}"
+TMP="$(mktemp -d "${TMPDIR:-/tmp}/w21-figures-build.XXXXXX")"
 PANELS=()
 [ -f "$HERE/data/cmazet_items.json" ] && PANELS=(--panels "$HERE/data/cmazet_items.json")
 mkdir -p "$TMP/a" "$TMP/b"
-REPO="$(dirname "$HERE")"
 for d in a b; do
-  python "$HERE/make_figures.py" --out "$TMP/$d"
-  python "$HERE/make_figures_w22.py" --out "$TMP/$d" ${PANELS[@]+"${PANELS[@]}"} --merge "$TMP/$d/PROVENANCE.json"
+  "$PY" "$HERE/make_figures.py" --out "$TMP/$d"
+  "$PY" "$HERE/make_figures_w22.py" --out "$TMP/$d" ${PANELS[@]+"${PANELS[@]}"} --merge "$TMP/$d/PROVENANCE.json"
   # w22 figures3 (AMENDMENT-1): fig_task_cmazet, drawn with the panels' own draw_state; the glyph
   # identity gate (task figure agent == result-panel agent) must pass before anything is installed
-  (cd "$REPO" && python -m w19_task_figures.make_task_figures --only cmazet \
-      --assets /Users/batmac/clu/outputs/w19-task-figures/assets --out "$TMP/task_$d")
+  "$PY" "$TASKFIG" --only cmazet --assets "$ASSETS" --out "$TMP/task_$d"
   mv "$TMP/task_$d/fig_task_cmazet.pdf" "$TMP/task_$d/fig_task_cmazet.png" "$TMP/$d/"
-  (cd "$REPO" && python -m w19_task_figures.verify_glyph "$TMP/$d" "$TMP/$d" "$TMP/$d/GLYPH-IDENTITY.json" > /dev/null)
-  python "$HERE/tools/merge_task_prov.py" "$TMP/$d/PROVENANCE.json" \
+  "$PY" "$VERIFY" "$TMP/$d" "$TMP/$d" "$TMP/$d/GLYPH-IDENTITY.json" > /dev/null
+  "$PY" "$MERGE" "$TMP/$d/PROVENANCE.json" \
       "$TMP/task_$d/PROVENANCE-task-figures.json" "$TMP/$d/GLYPH-IDENTITY.json"
   rm -r "$TMP/task_$d"
 done
-(cd "$TMP/a" && shasum -a 256 * | sort -k2) > "$TMP/sha_a.txt"
-(cd "$TMP/b" && shasum -a 256 * | sort -k2) > "$TMP/sha_b.txt"
+# manifests sorted in the C locale, so a fresh copy under any locale writes the same bytes
+(cd "$TMP/a" && shasum -a 256 * | LC_ALL=C sort -k2) > "$TMP/sha_a.txt"
+(cd "$TMP/b" && shasum -a 256 * | LC_ALL=C sort -k2) > "$TMP/sha_b.txt"
 if diff "$TMP/sha_a.txt" "$TMP/sha_b.txt"; then echo "REBUILD: BYTE-IDENTICAL"; else echo "REBUILD: DIFFERS"; exit 1; fi
+# the code a folder needs to rebuild itself (flat; figroot.py resolves the layout)
+SCRIPTS=("$HERE/make_figures.py" "$HERE/make_figures_w21.py" "$HERE/make_figures_w22.py"
+         "$HERE/maze_six.py" "$HERE/validate_palette.py" "$HERE/build.sh" "$HERE/figroot.py"
+         "$HERE/cmazet_panels.py" "$HERE/cmazet_door_search.py" "$HERE/cmazet_extract_rel.py"
+         "$HERE/cmazet_ambiguity.py" "$TASKFIG" "$VERIFY")
 for D in "$DEST" "$DEST2"; do
-  mkdir -p "$D" "$D/data"
+  mkdir -p "$D" "$D/data" "$D/tools"
   cp "$TMP"/a/fig_*.pdf "$TMP"/a/fig_*.png "$TMP/a/PROVENANCE.json" "$TMP/a/GLYPH-IDENTITY.json" "$D/"
-  cp "$HERE/make_figures.py" "$HERE/make_figures_w21.py" "$HERE/make_figures_w22.py" \
-     "$HERE/maze_six.py" "$HERE/validate_palette.py" "$HERE/build.sh" "$D/"
-  [ -f "$HERE/cmazet_panels.py" ] && cp "$HERE/cmazet_panels.py" "$D/"
-  cp "$HERE/cmazet_extract_rel.py" "$HERE/cmazet_ambiguity.py" "$REPO/w19_task_figures/make_task_figures.py" \
-     "$REPO/w19_task_figures/verify_glyph.py" "$D/"
-  cp "$HERE"/data/*.json "$D/data/"
+  if [ "$(cd "$D" && pwd -P)" != "$(cd "$HERE" && pwd -P)" ]; then   # in place: code is already here
+    cp "${SCRIPTS[@]}" "$D/"
+    cp "$MERGE" "$HERE/tools/vendor_engine.py" "$D/tools/"
+    cp "$HERE"/data/*.json "$D/data/"
+    "$PY" "$HERE/tools/vendor_engine.py" "$CODE" "$D"
+  fi
   cp "$TMP/sha_a.txt" "$D/SHA256-w22-build1.txt"
   cp "$TMP/sha_b.txt" "$D/SHA256-w22-build2.txt"
   # every file this build produced, re-hashed at its destination (other spokes' files are not ours)
-  (cd "$D" && awk '{print $2}' "$TMP/sha_a.txt" | xargs shasum -a 256 | sort -k2) > "$TMP/sha_inst.txt"
+  (cd "$D" && awk '{print $2}' "$TMP/sha_a.txt" | xargs shasum -a 256 | LC_ALL=C sort -k2) > "$TMP/sha_inst.txt"
   if diff "$TMP/sha_a.txt" "$TMP/sha_inst.txt"; then echo "INSTALLED BYTE-IDENTICAL: $D"
   else echo "INSTALL DIFFERS: $D"; exit 1; fi
 done
diff --git a/w21_figures/cmazet_extract_rel.py b/w21_figures/cmazet_extract_rel.py
index f035bbd..15868f3 100644
--- a/w21_figures/cmazet_extract_rel.py
+++ b/w21_figures/cmazet_extract_rel.py
@@ -25,8 +25,14 @@ import sys
 import time
 
 HERE = os.path.dirname(os.path.abspath(__file__))
-REPO = os.path.dirname(HERE)
+# PW-1 item 4: the re-read loads persisted models and needs a FULL CHLU checkout (the peer runner
+# and w9_harness.encoder are not in a figure folder's engine/): the repo this file sits in, or
+# $CHLU_FIG_CODE. Not part of build.sh; its output data/cmazet_items_rel.json is.
+REPO = os.environ.get("CHLU_FIG_CODE") or os.path.dirname(HERE)
 VEND = os.path.join(REPO, "peer_w21_clu_improve_e7fdb9b")
+if not os.path.isdir(VEND):
+    raise SystemExit("cmazet_extract_rel.py needs a full CHLU checkout with %s: set CHLU_FIG_CODE"
+                     % os.path.basename(VEND))
 for p in (VEND, os.path.join(REPO, "scripts"), REPO):
     if p not in sys.path:
         sys.path.insert(0, p)
diff --git a/w21_figures/cmazet_panels.py b/w21_figures/cmazet_panels.py
index 4a0a965..87ffd60 100644
--- a/w21_figures/cmazet_panels.py
+++ b/w21_figures/cmazet_panels.py
@@ -20,7 +20,9 @@ import sys
 import numpy as np
 
 HERE = os.path.dirname(os.path.abspath(__file__))
-REPO = os.path.dirname(HERE)
+if HERE not in sys.path:
+    sys.path.insert(0, HERE)
+import figroot  # noqa: E402  (PW-1 item 4: the engine root in the repo AND in a flat figure folder)
 
 VALID = ("✓", "#1f7a32")
 INVALID = ("✗", "#c0392b")
@@ -34,13 +36,12 @@ CELL = 10
 
 
 def _engine():
-    for p in (REPO, os.path.join(REPO, "scripts")):
-        if p not in sys.path:
-            sys.path.insert(0, p)
+    root = figroot.add_code_to_path()
     import chlu
-    assert os.path.dirname(os.path.dirname(os.path.abspath(chlu.__file__))) == REPO, chlu.__file__
+    assert os.path.dirname(os.path.dirname(os.path.abspath(chlu.__file__))) == root, chlu.__file__
     import chlu.exp_w15_cmaze_t as T
     import w16_cmaze_t as W16
+    figroot.check_loaded()
     return T, W16
 
 
diff --git a/w21_figures/figroot.py b/w21_figures/figroot.py
new file mode 100644
index 0000000..9b76a70
--- /dev/null
+++ b/w21_figures/figroot.py
@@ -0,0 +1,101 @@
+"""Where the figure build finds its code, in either of its two layouts (PW-1 item 4).
+
+Layout "repo":   this file sits in <CHLU repo>/w21_figures/; the task-figure scripts are in
+                 <repo>/w19_task_figures/ and the maze engine (chlu, w8_envscout, w9_harness,
+                 scripts/w16_cmaze_t.py) is the repo itself.
+Layout "folder": this file sits in a flat figure folder (outputs/paper-visuals/, w17-paper/figures/,
+                 or any copy of them); every script is next to it and the maze engine is the
+                 folder's own snapshot in engine/ (written by tools/vendor_engine.py, pinned by
+                 engine/ENGINE.json).
+The engine root is, in order: $CHLU_FIG_CODE if set; the repo (layout "repo"); ./engine/.
+Nothing here reads evidence; evidence stays at $CHLU_EVIDENCE (default ~/clu/evidence/outputs).
+usage (shell): python figroot.py layout|code|script <repo-relative path>|check
+"""
+import os
+import sys
+
+HERE = os.path.dirname(os.path.abspath(__file__))
+PARENT = os.path.dirname(HERE)
+ENGINE_PROBE = os.path.join("chlu", "exp_w15_cmaze_t.py")
+#: top-level modules that must come from the engine root (never from another checkout on sys.path,
+#: e.g. the venv's editable chlu install)
+ENGINE_TOPS = ("chlu", "w8_envscout", "w9_harness", "w15_cmaze_t_run", "w16_cmaze_t")
+
+
+def layout():
+    if os.path.basename(HERE) == "w21_figures" and os.path.isfile(os.path.join(PARENT, ENGINE_PROBE)):
+        return "repo"
+    return "folder"
+
+
+def code_root():
+    env = os.environ.get("CHLU_FIG_CODE")
+    cands = [("$CHLU_FIG_CODE", os.path.abspath(env))] if env else []
+    if layout() == "repo":
+        cands.append(("repo", PARENT))
+    cands.append(("engine/", os.path.join(HERE, "engine")))
+    for why, c in cands:
+        if os.path.isfile(os.path.join(c, ENGINE_PROBE)):
+            return c
+        if why == "$CHLU_FIG_CODE":
+            break
+    raise SystemExit(
+        "figroot: no maze engine found (tried %s). The CMAZE-T figures need the engine code: a "
+        "folder copy carries it in engine/ (README.md, 'Rebuild'); or set CHLU_FIG_CODE to a CHLU "
+        "checkout at f571b2b or later." % ", ".join("%s=%s" % c for c in cands))
+
+
+def script(rel):
+    """A build script named by its repo-relative path (as PROVENANCE.json records it)."""
+    if layout() == "repo":
+        return os.path.join(PARENT, rel)
+    return os.path.join(HERE, os.path.basename(rel))
+
+
+def engine_file(rel):
+    """An engine source named by its repo-relative path (chlu/..., w8_envscout/..., ...)."""
+    return os.path.join(code_root(), rel)
+
+
+def add_code_to_path():
+    root = code_root()
+    for p in (os.path.join(root, "scripts"), root):
+        if p not in sys.path:
+            sys.path.insert(0, p)
+    return root
+
+
+def check_loaded():
+    """Every loaded engine module comes from the engine root; returns the count checked."""
+    root = os.path.realpath(code_root()) + os.sep
+    n = 0
+    for name, mod in list(sys.modules.items()):
+        if name.split(".")[0] not in ENGINE_TOPS:
+            continue
+        f = getattr(mod, "__file__", None)
+        if not f:
+            continue
+        assert os.path.realpath(f).startswith(root), \
+            "engine module %s loaded from %s, not from %s" % (name, f, root)
+        n += 1
+    return n
+
+
+if __name__ == "__main__":
+    cmd = sys.argv[1] if len(sys.argv) > 1 else "code"
+    if cmd == "layout":
+        print(layout())
+    elif cmd == "code":
+        print(code_root())
+    elif cmd == "script":
+        print(script(sys.argv[2]))
+    elif cmd == "check":
+        add_code_to_path()
+        import chlu.exp_w15_cmaze_t  # noqa: F401
+        import w16_cmaze_t  # noqa: F401
+        import w8_envscout.maze_colour  # noqa: F401
+        import w9_harness.dynamics  # noqa: F401
+        print("engine OK: %s (%s layout), %d modules from it" % (code_root(), layout(),
+                                                                   check_loaded()))
+    else:
+        raise SystemExit(__doc__)
diff --git a/w21_figures/make_figures_w22.py b/w21_figures/make_figures_w22.py
index 547cc71..cf6db5e 100644
--- a/w21_figures/make_figures_w22.py
+++ b/w21_figures/make_figures_w22.py
@@ -5,8 +5,8 @@ check are used unchanged). Stems drawn here (each overwrites the pass-1 file of
   fig_crossover         unchanged from pass 1 (make_figures_w21.fig_crossover)
   fig_ladder            PW-1 item 10: ONE panel, the w20 131,072-row data-matched set; no title
                         (pool size and steps in the caption), legend at the right, no end labels
-  fig_headline_steps    NEW: rows 3-4 TEST, rival score against optimiser steps at three pool sizes
-                        (w20: the operative variable was steps, not data)
+  fig_headline_steps    retired from the build (no slot; retired-pw1/); the function is kept, not
+                        called (PW-1 item 4)
   fig_maze_conditions   PW-1 item 1: (a) four input conditions, every seed; (b) / (c) six seeds
                         without / with relational coding; no labels, ceiling, hard/easy or overlay
   fig_maze_rel_tuples   retired from the build (PW-1 item 15); the function is kept, not called
@@ -701,12 +701,12 @@ def main():
     os.makedirs(args.out, exist_ok=True)
     W.fig_crossover(args.out)
     fig_ladder(args.out)
-    fig_headline_steps(args.out)
     fig_maze_conditions(args.out)
     fig_deletion(args.out)
     if args.panels:
         import cmazet_panels  # noqa: E402
         cmazet_panels.build(args.out, args.panels, W.PROV, save_fn=save)
+        cmazet_panels.figroot.check_loaded()  # PW-1 item 4: every engine module from the engine root
     with open(args.merge, "r", encoding="utf-8") as f:
         base = json.load(f)
     for stem, e in base["figures"].items():
diff --git a/w21_figures/tools/merge_task_prov.py b/w21_figures/tools/merge_task_prov.py
index 2ca05df..b2237b1 100644
--- a/w21_figures/tools/merge_task_prov.py
+++ b/w21_figures/tools/merge_task_prov.py
@@ -6,11 +6,21 @@ import json
 import os
 import sys
 
-REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
+sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
+import figroot  # noqa: E402  (PW-1 item 4: resolves the sources in the repo or in a flat folder)
+
 SRC = ["w19_task_figures/make_task_figures.py", "w21_figures/cmazet_panels.py",
        "w8_envscout/maze_colour.py", "w9_harness/dynamics.py", "chlu/exp_w15_cmaze_t.py"]
 
 
+def _path(s):
+    """SRC names stay repo-relative (as PROVENANCE.json records them); the file is found in
+    either layout: build scripts via figroot.script, engine sources under the engine root."""
+    if s.split("/")[0] in ("w19_task_figures", "w21_figures"):
+        return figroot.script(s)
+    return figroot.engine_file(s)
+
+
 def main(prov_p, task_p, glyph_p):
     P = json.load(open(prov_p))
     t = json.load(open(task_p))["fig_task_cmazet"]
@@ -20,7 +30,7 @@ def main(prov_p, task_p, glyph_p):
     P["figures"]["fig_task_cmazet"] = dict(
         t, built_by="w19_task_figures/make_task_figures.py --only cmazet (w22 figures3, AMENDMENT-1)",
         banner=False, placeholder=False, awaiting=None,
-        sources=SRC, source_sha256={s: hashlib.sha256(open(os.path.join(REPO, s), "rb").read())
+        sources=SRC, source_sha256={s: hashlib.sha256(open(_path(s), "rb").read())
                                     .hexdigest() for s in SRC},
         notes="Task definition, no result. Every maze cell is drawn by the result panels' renderer "
               "cmazet_panels.draw_state (agent = black disc, centre = pocket colour, white = empty); "
diff --git a/w21_figures/tools/vendor_engine.py b/w21_figures/tools/vendor_engine.py
new file mode 100644
index 0000000..96ba8df
--- /dev/null
+++ b/w21_figures/tools/vendor_engine.py
@@ -0,0 +1,89 @@
+#!/usr/bin/env python3
+"""Snapshot the maze engine the figure build imports into <dest>/engine/ (PW-1 item 4).
+
+The CMAZE-T panels and the task figure re-derive every drawn maze from the deterministic engine
+(chlu.exp_w15_cmaze_t, scripts/w16_cmaze_t.py, w8_envscout, w9_harness). A flat figure folder is
+not a CHLU repo, so it carries its own copy of exactly the engine files the build loads: the import
+closure of those four modules, found by importing them in a clean subprocess and keeping every
+loaded file under the source root. Files are copied byte for byte; engine/ENGINE.json records the
+source commit and each file's SHA-256, and the copy is re-hashed against it.
+usage: python vendor_engine.py <CHLU repo root> <dest folder>      (writes <dest>/engine/)
+"""
+import hashlib
+import json
+import os
+import shutil
+import subprocess
+import sys
+
+ROOTS = ("chlu.exp_w15_cmaze_t", "w16_cmaze_t", "w8_envscout.maze_colour", "w9_harness.dynamics")
+PROBE = r"""
+import os, sys, json
+root = sys.argv[1]
+sys.path[:0] = [os.path.join(root, "scripts"), root]
+for m in %r:
+    __import__(m)
+out = set()
+for mod in list(sys.modules.values()):
+    f = getattr(mod, "__file__", None)
+    if f and os.path.realpath(f).startswith(os.path.realpath(root) + os.sep):
+        out.add(os.path.relpath(os.path.realpath(f), os.path.realpath(root)))
+print(json.dumps(sorted(out)))
+""" % (ROOTS,)
+
+
+def sha(p):
+    return hashlib.sha256(open(p, "rb").read()).hexdigest()
+
+
+def git(root, *a):
+    try:
+        return subprocess.run(["git", "-C", root] + list(a), capture_output=True, text=True,
+                              check=True).stdout.strip()
+    except (OSError, subprocess.CalledProcessError):
+        return None
+
+
+def main(root, dest):
+    root = os.path.abspath(root)
+    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
+    env["PYTHONDONTWRITEBYTECODE"] = "1"
+    rels = json.loads(subprocess.run([sys.executable, "-c", PROBE, root], capture_output=True,
+                                     text=True, check=True, env=env).stdout)
+    rels = [r for r in rels if r.endswith(".py")]
+    assert "chlu/exp_w15_cmaze_t.py" in rels, rels
+    eng = os.path.join(dest, "engine")
+    want = {r: sha(os.path.join(root, r)) for r in rels}
+    try:
+        have = json.load(open(os.path.join(eng, "ENGINE.json")))["files"]
+        if have == want and all(sha(os.path.join(eng, r)) == h for r, h in have.items()):
+            print("engine: unchanged, %d files -> %s" % (len(want), eng))
+            return
+    except (OSError, ValueError, KeyError):
+        pass
+    if os.path.exists(eng):
+        # never delete: move the old snapshot aside (the caller keeps or discards it)
+        os.rename(eng, eng + ".superseded-%d" % os.getpid())
+    files = {}
+    for r in rels:
+        os.makedirs(os.path.join(eng, os.path.dirname(r)), exist_ok=True)
+        shutil.copyfile(os.path.join(root, r), os.path.join(eng, r))
+        files[r] = sha(os.path.join(root, r))
+        assert sha(os.path.join(eng, r)) == files[r], r
+    head = git(root, "rev-parse", "HEAD")
+    dirty = git(root, "status", "--porcelain", "--", *rels)
+    rec = dict(what="maze-engine snapshot for the figure build (figroot.py resolves it)",
+               written_by="w21_figures/tools/vendor_engine.py", import_roots=list(ROOTS),
+               source_commit=head, source_clean=(dirty == "") if dirty is not None else None,
+               n_files=len(files), files=files)
+    with open(os.path.join(eng, "ENGINE.json"), "w", encoding="utf-8") as f:
+        json.dump(rec, f, indent=1, sort_keys=True)
+        f.write("\n")
+    print("engine: %d files from %s @ %s (clean=%s) -> %s" % (len(files), root, head,
+                                                              rec["source_clean"], eng))
+
+
+if __name__ == "__main__":
+    if len(sys.argv) != 3:
+        raise SystemExit(__doc__)
+    main(sys.argv[1], sys.argv[2])
```
</details>

## README top section (installed in both folders)
# paper-visuals — figure index

## Rebuild: the one command

```
cd ~/clu/outputs/paper-visuals && bash build.sh
```

This builds every figure this folder owns (below) twice from the evidence, stops unless the two
builds are byte-identical, then installs them into `~/clu/outputs/w17-paper/figures/` and
`~/clu/outputs/paper-visuals/` and re-hashes them there. The same command works in
`~/clu/outputs/w17-paper/figures/` (same code), in a fresh copy of either folder anywhere (e.g.
under `/tmp`), and from a CHLU checkout as `bash w21_figures/build.sh`. No `source` step is needed:
`build.sh` sets its own `PYTHONPATH`. It needs `python` from the project venv on `PATH`
(`~/clu/workspace/.venv/bin`), or `PYTHON=<that python> bash build.sh`.

- **Try an edit without touching the installed files:** `bash build.sh <scratch dir 1> <scratch dir 2>`.
- **Plot-edit loop:** edit `make_figures_w22.py`, `make_figures_w21.py`, `cmazet_panels.py` or a
  `data/*.json` file here, then run `bash build.sh`. The build copies its code only into a
  destination that is not this folder. To keep an edit, copy the file back to `w21_figures/` (or
  `w19_task_figures/`) in the CHLU repo and commit it there.
- **Built here** (10 stems: 20 files, plus `PROVENANCE.json`, `GLYPH-IDENTITY.json` and the
  manifests `SHA256-w22-build{1,2}.txt`): `fig_task_cmazet`, `fig_ladder`, `fig_crossover`,
  `fig_deletion`, `fig_encoder_adapt`, `fig_maze_conditions`, `fig_cmazet_families`,
  `fig_cmazet_per_arm`, `fig_cmazet_rel_per_arm` and `fig_cmazet_door_then_tint`.
  `fig_headline_steps` is no longer built. It has no slot, and its last build is in `retired-pw1/`.
- **Inputs it reads:**
  - the evidence tree `$CHLU_EVIDENCE` (default `~/clu/evidence/outputs`);
  - `data/*.json` in this folder;
  - the maze-engine snapshot `engine/`: 61 files copied from a CHLU checkout by
    `tools/vendor_engine.py`, each hash-pinned in `engine/ENGINE.json` with the source commit
    (the files are byte-identical at `f571b2b` and `a01bea7`).

  Set `CHLU_FIG_CODE=<a CHLU checkout>` to use a checkout's engine instead. If no engine is found,
  the build stops before drawing and names exactly this. Every engine module is checked to load
  from the engine root, never from another checkout on `sys.path`.

### Mirrored figures (not built by `build.sh`): one rebuild line each

- `fig_task_shapes.{pdf,png}` (w19 task figures), rebuilt from the frozen w19 assets. The line
  below reproduces the installed bytes (`c83b24d5…` / `78a8d7ed…`, two builds, checked 2026-09-25):
  `cd ~/clu/outputs/paper-visuals && PYTHONPATH=.:engine/scripts:engine python make_task_figures.py --only all --assets ~/clu/outputs/w19-task-figures/assets --out /tmp/task-shapes && cp /tmp/task-shapes/fig_task_shapes.* .`
  The assets come from two upstream stages, run from a CHLU checkout:
  - `shapes_images.npz` (`9f9d71bc…`): `w19_task_figures/extract_images.py --h5 ~/clu/data/gen/shapes3d/3dshapes.h5`,
    using the Shapes3D dataset (sha256 `0a0f6ed9…`, the runbook pin) in the h5py venv
    `~/clu/outputs/w19-task-figures/h5venv`.
  - `task_assets.npz` (`bc6df90e…`): `w19_task_figures/extract_assets.py`, in the project venv (JAX)
    with the frozen decoder `~/clu/stage/outputs/gen-decoder-lift/winner/decoder_v1_s0.pkl`
    (param hash `01ef91f7…`) and the judge classifier (`c8741ac4…`). Note that `extract_assets.py`
    reads `w17-paper/figures/rendered/REQUESTS.json`, which is now under `rendered/retired-pw1/`.
- `rendered/rc_pw1_shapes_{per_model,baseline_wins}.{pdf,png}`, with `rendered/REQUESTS-pw1.json`
  and `rendered/PROVENANCE-render.json`, rendered through the one frozen decoder:
  `cd <CHLU checkout at a01bea7 or later> && bash pw1_render_shapes/build_twice.sh`
  This builds twice into `~/clu/outputs/pw1-render-shapes/build/{a,b}` (earlier builds are moved
  aside, not deleted). Then copy `build/a/rc_pw1_*`, `REQUESTS-pw1.json`, `PROVENANCE-render.json`
  and `SHA256-build{1,2}.txt` (as `SHA256-pw1-render-build{1,2}.txt`) into `rendered/`. The inputs
  are the project venv (JAX) and `pw1_render_shapes/run_env.sh`'s stage roots:
  - the frozen decoder `stage/outputs/gen-decoder-lift/winner` (`01ef91f7…`);
  - the judge classifier `stage/outputs/gen-external-classifier` (`c8741ac4…`);
  - the Shapes3D value hold-out split `stage/outputs/data/gen`;
  - the w20 baseline objects `stage/w20-shapes-data-matched`, hash-checked against `SHA256-w20-test-objects.txt`;
  - the n-CLU object `stage/outputs/w12-single-design-crossover/scored/run1_seed0_K32/arm.pkl`.

