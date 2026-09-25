"""Prepend (or replace) the PW-1 section of FIGURES.md in a figure folder. usage: figures_md.py FOLDER INDEX.json BUILD_COMMIT

The section lists every figure PW-1 changed, added or retired, with its plan item, the spoke branch that
changed its code, its label and figure number, and its installed SHA-256 (from the make_index.py INDEX json).
Earlier sections are kept below it, unchanged.
"""
import json, os, sys

PV, IDX, COMMIT = sys.argv[1:4]
idx = json.load(open(IDX))
rows = {r["file"]: r for r in idx["rows"]}
CHANGES = [
    ("fig_maze_conditions", "PW-1 items 1, 8", "pw1/plot-maze", "redrawn: four rows (words full, observation, words + observation for n-CLU and n-CLU-REL), every seed drawn, no claim labels, no ceiling, (c) relational comparison only; new caption"),
    ("fig_cmazet_families", "PW-1 item 16", "pw1/plot-maze", "no bottom paragraph (read once in Sec. maze-wm); plain names; + most-baselines-wrong row and a door success row (n-CLU-REL run, seed 2), drawn by rule, seed 16"),
    ("fig_cmazet_per_arm", "PW-1 item 16", "pw1/plot-maze", "no bottom paragraph; plain names; no door success exists for the n-CLU (stated in the caption)"),
    ("fig_cmazet_rel_per_arm", "PW-1 item 16", "pw1/plot-maze", "no bottom paragraph; plain names; n-CLU-REL success row is a door success drawn by rule"),
    ("fig_cmazet_door_then_tint", "PW-1 item 16", "pw1/plot-maze", "no bottom paragraph; + three n-CLU-REL door successes (seed 2) drawn by rule"),
    ("fig_ladder", "PW-1 item 10", "pw1/plot-shapes", "single data-matched panel, no title, legend right, no line-end labels, new names and y label; new caption"),
    ("fig_crossover", "PW-1 item 11", "pw1/plot-shapes", "axis labels, panel (b) title \"Memory footprint vs K\", legend n-CLU; new caption"),
    ("fig_deletion", "PW-1 items 12, 13", "pw1/plot-shapes", "only former (e), (f) remain as (a), (b), retitled; red sphere; the removed panels' numbers are in Tab. del and one sentence"),
    ("fig_encoder_adapt", "PW-1 items 2, 3", "pw1/plot-shapes", "y label in the new vocabulary; caption matches it"),
    ("rendered/rc_pw1_shapes_per_model", "PW-1 item 9", "pw1/render-shapes", "NEW: per-model success/failure grid against the w20 data-matched baselines, seed 0; Sec. shapes-t7"),
    ("rendered/rc_pw1_shapes_baseline_wins", "PW-1 item 9", "pw1/render-shapes", "NEW: baseline succeeds, n-CLU fails, one row per baseline, seed 0; Sec. shapes-t7"),
]
L = ["<!-- PW1-SECTION-START -->", "## ⭐ PW-1 figure install (pw1-draft-figures, 2026-09-25): the current state. Read this section first.", "",
     "Built by `w21_figures/build.sh` at `%s` (two builds byte-identical, each equal to the figure spokes' own builds), "
     "installed into `w17-paper/figures/` and `paper-visuals/`; rendered figures from pw1 render-shapes' build. "
     "Captions and slots in both drafts by pw1-draft-figures. The full per-file index is `README.md`." % COMMIT, "",
     "| figure | item | code branch | what changed | label, figure no. | installed SHA-256 (pdf) |", "|---|---|---|---|---|---|"]
for stem, item, br, what in CHANGES:
    r = rows[stem + ".pdf"]
    lab = r["placement"].split(",")[0] + ", " + r["placement"].split(",")[1].strip()
    L.append("| `%s` | %s | `%s` | %s | %s | `%s` |" % (stem, item, br, what, lab, r["sha256"]))
L += ["", "**Retired (moved into `retired-pw1/`, never deleted):** " + ", ".join(sorted({"`%s`" % os.path.splitext(r["file"])[0] for r in idx["retired"]})) + ".",
      "", "Unchanged and placed: `fig_task_shapes` (w19 mirror), `fig_task_cmazet` (built, unchanged bytes).", "",
      "Sections below are history (w22, w21, w17); where they disagree with this section or README.md, this section wins.",
      "<!-- PW1-SECTION-END -->", ""]
p = os.path.join(PV, "FIGURES.md")
old = open(p).read()
if "<!-- PW1-SECTION-START -->" in old:
    a = old.index("<!-- PW1-SECTION-START -->"); b = old.index("<!-- PW1-SECTION-END -->") + len("<!-- PW1-SECTION-END -->\n\n")
    old = old[:a] + old[b:]
head, rest = old.split("\n", 1) if old.startswith("# ") else ("# FIGURES.md", old)
open(p, "w").write(head + "\n\n" + "\n".join(L) + "\n" + rest.lstrip("\n"))
print("FIGURES.md PW-1 section written:", p)
