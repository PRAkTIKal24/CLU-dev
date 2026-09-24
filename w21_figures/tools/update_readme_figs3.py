#!/usr/bin/env python3
"""w22 figures3: refresh paper-visuals/README.md for the figures this pass changed or added.

Rows rewritten (both .pdf and .png): fig_task_cmazet, fig_cmazet_families, fig_cmazet_per_arm,
fig_cmazet_door_then_tint, fig_cmazet_deletion; rows ADDED: fig_cmazet_rel_per_arm. For each: the
figure number and page from the compiled DRAFT-v2.aux, the \\figslot line in each draft, the build
hash from SHA256-w22-build1.txt. The section text and "cited in text from" of an existing row are
kept (the sections did not change); the new row's are computed (label never \\ref'd). Rows of other
figures are NOT touched; stems the README lists but the current draft no longer places are recorded
in a dated note, not fixed (another session's change).
usage: update_readme_figs3.py <paper-visuals dir> <w17-paper dir>
"""
import os
import re
import sys

PV, W17 = sys.argv[1], sys.argv[2]
RD = os.path.join(PV, "README.md")
t = open(RD).read()
sha = dict(reversed(l.split()) for l in open(os.path.join(PV, "SHA256-w22-build1.txt")) if l.strip())
aux = open(os.path.join(W17, "DRAFT-v2.aux")).read()
d1 = open(os.path.join(W17, "DRAFT-v2.tex")).read().split("\n")
d2 = open(os.path.join(W17, "DRAFT-v2-style.tex")).read().split("\n")


def fignum(label):
    m = re.search(r"\\newlabel\{%s\}\{\{(\d+)\}\{(\d+)\}" % re.escape(label), aux)
    return m.group(1), m.group(2)


def slot(lines, stem):
    hits = [i + 1 for i, l in enumerate(lines) if l.startswith("\\figslot{%s}" % stem)]
    assert len(hits) == 1, (stem, hits)
    return hits[0]


BUILD = "(this folder's build (w22 figures3))"
STRIP = (" Under each predicted state a strip prints the four facts the method emitted (legal or illegal, "
         "move, key event, pocket after the move); the tick means all four are right.")
ROWS = {
    "fig_task_cmazet": dict(
        label="fig:task-cmazet",
        desc="The maze task, with no results: (a) one generated maze with coloured keys (diamonds), doors "
             "and tinted floor, the goal (a cross) and the agent, a black disc whose centre shows the key "
             "it holds (white = empty): the same symbols, drawn by the same code (cmazet_panels.draw_state), "
             "as every maze result panel; (b-d) three single moves (legal; blocked because the floor matches "
             "the key being held; legal because a matching door uses the key up first), each written as a "
             "five-word request and a four-fact answer. No step or budget line.",
        src="CMAZE-T engine (w8 maze_colour + w9 dynamics.step + chlu/exp_w15_cmaze_t.py), seed-0 TRAIN-bank "
            "maze, generator seed 2000006; drawn by w21_figures/cmazet_panels.draw_state via "
            "w19_task_figures/make_task_figures.py --only cmazet; PROVENANCE.json fig_task_cmazet; "
            "GLYPH-IDENTITY.json (agent glyph == result-panel glyph, pixel-equal)"),
    "fig_cmazet_families": dict(label="fig:cmazet-families", append=STRIP + (
        " In the seed-2 rule-a row the CLU (legal, marked wrong) and the MLP (illegal, marked right) "
        "have the same picture and are told apart by the strip.")),
    "fig_cmazet_per_arm": dict(label="fig:cmazet-per-arm", append=STRIP),
    "fig_cmazet_door_then_tint": dict(label="fig:cmazet-door-tint", append=STRIP),
    "fig_cmazet_deletion": dict(label="fig:cmazet-deletion", append=STRIP),
}
NEW = dict(
    stem="fig_cmazet_rel_per_arm", label="fig:cmazet-rel-per-arm", after="fig_cmazet_per_arm",
    desc="Maze task, CLU with colour-relational request coding (CLU-REL-A10), seed 0: for each of 19 "
         "methods (the CLU, six matched rivals, four controls, two blind arms, and six rivals given the "
         "CLU's relational request columns), one held-out move it predicts exactly and one it gets "
         "wrong." + STRIP,
    src="stage/rel-persist/{w21-maze-t-clu-improve,w21-rel-newseeds} persisted objects (hash-checked); "
        "evidence/outputs/ w21-maze-t-clu-improve/test/BOTH_test_seed{0,1,2}.json, "
        "w21-rel-newseeds/test/BOTH_test_seed{3,4,5}.json; data/cmazet_items_rel.json (gate 2: "
        "data/cmazet_items_rel.REPRO.json, 132/132 arm scores float-equal)")

WHERE = re.compile(r"Figure \d+, `(fig:[a-z-]+)`, (§[^|]*?) \(p\. \d+\); slot at DRAFT-v2:\d+, "
                   r"DRAFT-v2-style:\d+; cited in text from: ([^|]*?) \|")


def where(stem, label, section, cited):
    n, p = fignum(label)
    return "Figure %s, `%s`, %s (p. %s); slot at DRAFT-v2:%d, DRAFT-v2-style:%d; cited in text from: %s |" % (
        n, label, section, p, slot(d1, stem), slot(d2, stem), cited)


out = []
sections = {}
for line in t.split("\n"):
    m = re.match(r"\| `(fig_[a-z_]+)\.(pdf|png)` \| (.*)$", line)
    if m and m.group(1) in ROWS:
        stem, ext = m.group(1), m.group(2)
        R = ROWS[stem]
        cells = line.split(" | ")
        w = WHERE.search(line)
        assert w and w.group(1) == R["label"], (stem, line[:200])
        sections[stem] = (w.group(2), w.group(3))
        desc = R.get("desc") or (cells[1] + R["append"])
        src = R.get("src") or cells[3]
        line = "| `%s.%s` | %s | %s %s | `%s` %s |" % (
            stem, ext, desc, where(stem, R["label"], w.group(2), w.group(3).strip()), src,
            sha["%s.%s" % (stem, ext)], BUILD)
    out.append(line)
    if m and m.group(1) == NEW["after"] and m.group(2) == "png":
        sec = sections[NEW["after"]][0]
        for ext in ("pdf", "png"):
            out.append("| `%s.%s` | %s | %s %s | `%s` %s |" % (
                NEW["stem"], ext, NEW["desc"],
                where(NEW["stem"], NEW["label"], sec, "nowhere (placed, never \\ref'd)"), NEW["src"],
                sha["%s.%s" % (NEW["stem"], ext)], BUILD))
t = "\n".join(out)

old_pending = re.search(r"- \*\*Pending slot \(not a file, not placed by any draft\):\*\* `fig_cmazet_rel_per_arm`[^\n]*", t)
assert old_pending, "pending-slot paragraph not found"
t = t.replace(old_pending.group(0),
              "- **`fig_cmazet_rel_per_arm` is now drawn and placed** (w22 figures3, Figure 12, "
              "`fig:cmazet-rel-per-arm`): gate 2 passed on this machine with `w21_clu` from the "
              "~/clu/ship/ bundles; 132/132 arm scores re-read float-equal to the scored run JSONs, "
              "seeds 0-5 (data/cmazet_items_rel.REPRO.json).")

draft_stems = set(re.findall(r"\\figslot\{([^}]+)\}", "\n".join(d1)))
listed = set(s.replace("rendered/", "") for s in re.findall(r"^\| `([^`]+)\.pdf`", t, re.M))
gone = sorted(listed - draft_stems)
note = ("\n> **w22 figures3 (2026-09-25) refresh note.** Rows for fig_task_cmazet, fig_cmazet_families, "
        "fig_cmazet_per_arm, fig_cmazet_door_then_tint and fig_cmazet_deletion were rewritten and "
        "fig_cmazet_rel_per_arm was added; their figure numbers, pages and slot lines are read from the "
        "current DRAFT-v2.aux and both drafts. **Other rows were not regenerated and their \"where used\" "
        "may be stale**: the current drafts place %d figures, and these stems listed below are no longer "
        "placed by DRAFT-v2: %s. The completeness counts below predate this refresh.\n"
        % (len(draft_stems), ", ".join("`%s`" % g for g in gone)))
t = t.replace("\n## Completeness, both ways\n", note + "\n## Completeness, both ways\n", 1)

files_old = "| `README.md` |"
files_new = ("| `GLYPH-IDENTITY.json` | w22 figures3: proof that fig_task_cmazet's agent glyph is the result "
             "panels' glyph (disc pixels equal for pockets c1, c3 and empty, incl. seed 2 item 482; ring "
             "and centre colours present as the same bytes in both PDFs). Written by the build. |\n"
             "| `data/cmazet_items_rel.json, data/cmazet_items_rel.REPRO.json` | Per-item re-read of the "
             "CLU-REL-A10 persisted objects (cmazet_extract_rel.py) and its gate-2 reproduction check. |\n"
             "| `data/CMAZET-AMBIGUITY.json` | Same-picture/opposite-mark pair counts before and after the "
             "fact strip (cmazet_ambiguity.py); 0 after, in every scope. |\n"
             "| `cmazet_extract_rel.py, cmazet_ambiguity.py, make_task_figures.py, verify_glyph.py` | w22 "
             "figures3 code: REL gate-2 reader; the ambiguity counter; the task figure (w19_task_figures, "
             "run with --only cmazet); the glyph-identity check. |\n| `README.md` |")
assert t.count(files_old) == 1
t = t.replace(files_old, files_new)
t = t.replace("| `SHA256-mirrored-w19-task-figures-build{1,2}.txt` | Copied here from "
              "w19-task-figures/proof/SHA256-build{1,2}.txt: the manifest for fig_task_* (built by w19 "
              "task-figures, not by this folder's build). |",
              "| `SHA256-mirrored-w19-task-figures-build{1,2}.txt` | Copied here from "
              "w19-task-figures/proof/SHA256-build{1,2}.txt: now valid for fig_task_shapes only. "
              "fig_task_cmazet is built by this folder's build since w22 figures3 (hash in "
              "SHA256-w22-build1.txt). |")
open(RD, "w").write(t)
print("README updated; stems no longer placed:", gone)
