"""PW-1 figure index: README.md for a figure folder (w17-paper/figures/ or paper-visuals/), from artefacts.

Port of w22_paper_visuals/make_index.py (branch w22/paper-visuals; the original is kept beside this file as
make_index_w22_orig.py). Changes: the folder, the drafts, the .aux directory and the report directory are
arguments (the original wrote into outputs/w22-paper-visuals/, which a PW spoke may not write); placements
are read from \\figslot calls AND from figure environments with \\includegraphics (the rendered Shapes3D
figures); files in retired-pw1/ are listed as retired, not as unreferenced; manifests are the pw1 build's.

usage: python make_index.py FOLDER PAPER_DIR AUX_DIR OUT_DIR BUILD_COMMIT
Reads (never writes) the drafts, their compiled .aux, the folder's manifests and PROVENANCE.json.
Writes FOLDER/README.md and OUT_DIR/INDEX-<folder>.json. Exit 1 on a missing file, a hash problem or a
PNG/PDF parity failure. Every field except "what it shows" is derived here; DESCRIPTIONS were written
after opening each PNG of the installed build.
"""
import hashlib
import json
import os
import re
import sys

PV, PAPER, AUX, OUT, COMMIT = sys.argv[1:6]
# PW-1 item 4: the README's Rebuild section and the build-code bullet live in one place
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "pw1_visuals_tooling"))
import readme_block  # noqa: E402
DRAFTS = ["DRAFT-v2.tex", "DRAFT-v2-style.tex"]
NAME = "paper-visuals" if PV.rstrip("/").endswith("paper-visuals") else "w17-paper-figures"
W19TASK = "/Users/batmac/clu/outputs/w19-task-figures/proof"

# (manifest build 1, manifest build 2, path prefix, who built it)
MANIFESTS = [
    (os.path.join(PV, "SHA256-w22-build1.txt"), os.path.join(PV, "SHA256-w22-build2.txt"), "",
     "this folder's build.sh (w21_figures/build.sh at %s)" % COMMIT[:7]),
    (os.path.join(W19TASK, "SHA256-build1.txt"), os.path.join(W19TASK, "SHA256-build2.txt"), "",
     "w19 task-figures (mirrored)"),
    (os.path.join(PV, "rendered/SHA256-pw1-render-build1.txt"), os.path.join(PV, "rendered/SHA256-pw1-render-build2.txt"),
     "rendered/", "pw1 render-shapes (pw1_render_shapes/build.py, mirrored)"),
]

DESCRIPTIONS = {
    "fig_task_shapes": "The image task, with no results: (a) the six things that vary between images (floor, wall and object colour, object size, shape, viewing angle); (b) a request (six values) turned into an image by the fixed decoder and read back by a separate classifier; (c, d) which value combinations are kept out of training.",
    "fig_task_cmazet": "The maze task, with no results: (a) one generated maze with coloured keys, doors and tinted floor, the agent and the goal; (b-d) three single moves (legal; blocked because the floor matches the key being held; legal because a matching door uses the key up first), each written as a five-word request and a four-fact answer.",
    "rendered/rc_pw1_shapes_per_model": "Image task, seed 0, K*=32: for each of the five models (n-CLU, MLP, LSTM, Transformer, Continuous MLP) one held-out request it renders validly and one it renders invalidly (10 rows), with every model's image for that request and the lookup table's as a sixth column, each marked with the judge's tick or cross. The requested factor values are printed left of each row; the drawn model's panel is framed.",
    "rendered/rc_pw1_shapes_baseline_wins": "Image task, seed 0: one held-out request per baseline (MLP, LSTM, Transformer, Continuous MLP) on which that baseline's image is valid and the n-CLU's is not, every model's image side by side with the judge's marks; the n-CLU's and the baseline's panels are framed. Shapes drawn: sphere, cylinder, capsule, sphere.",
    "fig_ladder": "One panel, no title: distinct valid novel tuples per held-out combination against n, the number of factors jointly held out (2 to 6), for the n-CLU and four data-matched baselines (MLP, LSTM, Transformer, Continuous MLP) fitted on each rung's own 131,072 rows, with the lookup table as the floor; open markers are seeds, lines their mean, filled markers the seeds carried at 10x steps. Legend at the right. The n-CLU is on top at every n.",
    "fig_crossover": "(a) Validation composition against K (5 to 32 wells), the n-CLU's mean of 3 seeds with a one-standard-deviation band and each seed, and the highest of four controls at zero; (b) memory footprint vs K in bytes (x10^6), stacked in four parts (atoms, covariance + count, address book, coupling / hole table), with a 524,288-byte reference line.",
    "fig_deletion": "Two panels. (a) Composition on the surviving held-out combinations after minus before deletion, per seed with 95% bootstrap intervals, for deleting the value red, the red sphere and a random 1% of items (control): red sphere drops about 6, the other two stay near zero. (b) The n-CLU after deletion minus each baseline retrained without the same rows (MLP, LSTM, Transformer, Continuous MLP), per seed with 95% intervals; all above zero except three red-sphere intervals that reach zero.",
    "fig_encoder_adapt": "The n-CLU's margin over the 180k-parameter MLP baseline (novel tuples per held-out combination), per seed (lines join a seed), for the frozen encoder, an adapter and an anchor term, K*=32, 3 seeds.",
    "fig_maze_conditions": "Maze task, unseen n-way validity under the engine's judge. (a) One row per input choice (words full design, observation, words + observation for n-CLU and for n-CLU-REL): every seed of the n-CLU (circles) and of the row's best matched baseline (diamonds; Transformer-REL, hollow, in the last row), with the mean as a tick; 3 or 6 seeds per row, printed. (b) Seeds 0-5 without relational coding: n-CLU, Transformer (best baseline), highest control. (c) The same with relational coding: n-CLU-REL, Transformer-REL, highest control. No labels or overlays.",
    "fig_cmazet_families": "Maze task, seven held-out moves as rows (one per rule family at seed 0, rule A at seed 2; one move where the n-CLU is right and at least 4 of the 6 matched baselines are wrong; one door success from the n-CLU-REL run at seed 2, below a rule), columns: start state and move, the engine's next state, and 14 models' next states (n-CLU, matched baselines at 8,000 and 80,000 steps, four controls, two blind arms), each with a tick or cross and a two-line fact strip.",
    "fig_cmazet_per_arm": "Maze task, seed 0: for every model (n-CLU, six matched baselines, four controls, two blind arms), one held-out move it predicts correctly and one it gets wrong, in three columns of rows (start, engine's next state, the model's next state with tick or cross and fact strip). Two models have no correct move and show a failure only.",
    "fig_cmazet_rel_per_arm": "Maze task with relational coding: as fig_cmazet_per_arm for the n-CLU-REL, the matched baselines, controls and blind arms of the same runs, and MLP-REL, LSTM-REL and Transformer-REL at 8,000 and 80,000 steps; the n-CLU-REL success row is a door success at seed 2.",
    "fig_cmazet_door_then_tint": "Maze task, door-then-tint moves (a door opens with the key and the move enters a tint of the same colour): rows 1-3 at seed 0 (north, south, west), where the n-CLU is wrong; rows 4-6 door successes from the n-CLU-REL run at seed 2 (east, south, west), below a rule; every model's next state with tick or cross and fact strip.",
    # retired (in retired-pw1/)
    "fig_headline_steps": readme_block.RETIRED_HEADLINE,
    "fig_maze_readverb": "Retired (PW-1 item 14): route-task read-verb figure.",
    "fig_maze_requestlayer": "Retired (PW-1 item 14): request-layer audit figure.",
    "fig_maze_rel_tuples": "Retired (PW-1 item 15): per-held-out-tuple relational heat maps.",
    "fig_cmazet_deletion": "Retired (PW-1 item 17): CMAZE-T post-deletion panels.",
    "fig_maze_wm": "Retired (w18 maze world-model figure, never placed).",
}

MIRROR_SOURCES = {
    "fig_task_shapes": "w19-task-figures/proof/PROVENANCE-task-figures.json (3dshapes.h5 at the runbook pin; value_holdout/{factor_index.npz,value_cells.json}; frozen decoder seed 0 01ef91f7…; classifier c8741ac4…)",
    "rendered/rc_pw1_shapes_per_model": "rendered/PROVENANCE-render.json and rendered/REQUESTS-pw1.json (pw1-render-shapes): w20-shapes-data-matched objects (116/116 hash-checked), n-CLU w12-single-design-crossover run1_seed0_K32, frozen decoder 01ef91f7…, judge classifier c8741ac4…; seed 0, selection seed 20260925",
    "rendered/rc_pw1_shapes_baseline_wins": "rendered/PROVENANCE-render.json and rendered/REQUESTS-pw1.json (pw1-render-shapes), as rc_pw1_shapes_per_model",
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_manifest(p, prefix):
    m = {}
    for line in open(p):
        if line.strip():
            h, name = line.split(None, 1)
            m[prefix + name.strip()] = h
    return m


def strip_comment(line):
    return re.sub(r"(?<!\\)%.*", "", line)


def aux_labels(draft):
    txt = open(os.path.join(AUX, draft.replace(".tex", ".aux"))).read()
    return {lab: (num, page) for lab, num, page in
            re.findall(r"\\newlabel\{((?:fig|sec|app):[^}]*)\}\{\{([^}]*)\}\{([^}]*)\}", txt)}


def placements(draft):
    """stem -> label, line, figure number, page, section label; from \\figslot and figure/includegraphics."""
    src = open(os.path.join(PAPER, draft)).read().splitlines()
    labs = aux_labels(draft)
    code = [strip_comment(l) for l in src]
    out, refs, cur = {}, {}, None
    i = 0
    while i < len(code):
        line = code[i]
        m = re.search(r"\\label\{((?:sec|app):[^}]*)\}", line)
        if m:
            cur = m.group(1)
        for r in re.findall(r"\\ref\{(fig:[^}]*)\}", line):
            refs.setdefault(r, [])
            if cur not in refs[r]:
                refs[r].append(cur)
        if "\\figslot{" in line and "newcommand" not in line:
            j, buf = i, ""
            while True:
                buf += code[j] + "\n"
                lab = re.search(r"\}\{(fig:[^}]*)\}\s*$", buf.rstrip())
                if lab:
                    break
                j += 1
            stem = re.match(r"\s*\\figslot\{([^}]*)\}", buf).group(1)
            out[stem] = dict(label=lab.group(1), line=i + 1, section=cur)
        if "\\begin{figure}" in line:
            j, buf = i, ""
            while "\\end{figure}" not in code[j]:
                buf += code[j] + "\n"
                j += 1
            g = re.search(r"\\includegraphics(\[[^]]*\])?\{figures/([^}]*)\.pdf\}", buf)
            lab = re.search(r"\\label\{(fig:[^}]*)\}", buf)
            if g and lab:
                out[g.group(2)] = dict(label=lab.group(1), line=i + 1, section=cur)
        i += 1
    for v in out.values():
        num, page = labs[v["label"]]
        v["fig"], v["page"] = num, page
        s = labs.get(v["section"], ("?", "?"))
        v["section_num"] = s[0]
        v["cited_from"] = refs.get(v["label"], [])
    return out


def main():
    place = {d: placements(d) for d in DRAFTS}
    prov = json.load(open(os.path.join(PV, "PROVENANCE.json")))["figures"]
    man = {}
    for m1, m2, pre, who in MANIFESTS:
        a, b = read_manifest(m1, pre), read_manifest(m2, pre)
        assert a == b, ("manifest builds differ", m1)
        for n, h in a.items():
            man.setdefault(n, (h, who))
    files, retired = [], []
    for root, dirs, fs in os.walk(PV):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in fs:
            if f.endswith((".png", ".pdf")):
                rel = os.path.relpath(os.path.join(root, f), PV)
                (retired if "retired-pw1" in rel.split(os.sep) else files).append(rel)
    files.sort(), retired.sort()
    stems = sorted({os.path.splitext(f)[0] for f in files})
    referenced = set(place[DRAFTS[0]]) | set(place[DRAFTS[1]])
    missing = [s + e for s in sorted(referenced) for e in (".pdf", ".png") if not os.path.exists(os.path.join(PV, s + e))]
    unref = [f for f in files if os.path.splitext(f)[0] not in referenced]
    parity = [s for s in stems if not (os.path.exists(os.path.join(PV, s + ".pdf")) and os.path.exists(os.path.join(PV, s + ".png")))]
    same_place = all(place[DRAFTS[0]][s]["label"] == place[DRAFTS[1]][s]["label"] and
                     place[DRAFTS[0]][s]["fig"] == place[DRAFTS[1]][s]["fig"] for s in referenced)
    page_diff = max(abs(int(place[DRAFTS[0]][x]["page"]) - int(place[DRAFTS[1]][x]["page"])) for x in referenced)
    rows, problems = [], []
    for f in files:
        stem = os.path.splitext(f)[0]
        installed = sha(os.path.join(PV, f))
        mh, who = man.get(f, (None, None))
        if mh is None:
            problems.append("NO MANIFEST HASH: " + f)
        elif mh != installed:
            problems.append("HASH MISMATCH: %s installed %s manifest %s" % (f, installed, mh))
        if stem in prov:
            s = prov[stem]["sources"]
            s = s if isinstance(s, list) else [str(s)]
            if len(s) > 4:
                roots = sorted({"/".join(x.split("/")[:2]) for x in s})
                src = "%d files under: %s (full list: PROVENANCE.json figures.%s.sources)" % (len(s), "; ".join(roots), stem)
            else:
                src = "; ".join(s)
            src = "evidence/outputs/ " + src
        else:
            src = MIRROR_SOURCES.get(stem)
            if src is None:
                problems.append("NO DATA SOURCE: " + f)
                src = "?"
        if stem in referenced:
            p = place[DRAFTS[0]][stem]
            where = "Figure %s, `%s`, section %s `%s` (p. %s)" % (p["fig"], p["label"], p["section_num"], p["section"], p["page"])
            where += "; slot at " + ", ".join(d.replace(".tex", "") + ":%d" % place[d][stem]["line"] for d in DRAFTS)
            where += "; cited in text from: " + (", ".join(c for c in p["cited_from"] if c) or "nowhere (placed, never \\ref'd)")
        else:
            where = "**UNREFERENCED** by both drafts"
        desc = DESCRIPTIONS.get(stem)
        if desc is None:
            problems.append("NO DESCRIPTION: " + f)
            desc = "?"
        rows.append(dict(file=f, shows=desc, placement=where, source=src, sha256=installed,
                         manifest_sha256=mh, built_by=who, referenced=stem in referenced,
                         fig=int(place[DRAFTS[0]][stem]["fig"]) if stem in referenced else 99))
    ret_rows = []
    for f in retired:
        stem = os.path.splitext(os.path.basename(f))[0]
        ret_rows.append(dict(file=f, shows=DESCRIPTIONS.get(stem, "retired"), sha256=sha(os.path.join(PV, f)),
                             placed=stem in referenced or ("rendered/" + stem) in referenced))
        if ret_rows[-1]["placed"]:
            problems.append("RETIRED BUT PLACED: " + f)
    counts = dict(folder=PV, build_commit=COMMIT, drafts=DRAFTS,
                  include_targets_per_draft={d: len(place[d]) for d in DRAFTS},
                  include_targets_union=len(referenced), missing_files=missing,
                  files_here_png_pdf=len(files), files_here_referenced=len(files) - len(unref),
                  files_here_unreferenced=unref, retired_files=len(retired), stems_here=len(stems),
                  png_pdf_parity_failures=parity, placement_identical_in_both_drafts=same_place,
                  max_page_difference_between_drafts=page_diff, hash_problems=problems)
    json.dump(dict(counts=counts, rows=rows, retired=ret_rows), open(os.path.join(OUT, "INDEX-%s.json" % NAME), "w"), indent=1)
    return rows, ret_rows, counts


def md(t):
    return str(t).replace("|", "\\|").replace("\n", " ")


def write_readme(rows, ret_rows, c):
    L = ["# %s — figure index\n" % ("paper-visuals" if NAME == "paper-visuals" else "w17-paper/figures")]
    L.append(readme_block.REBUILD_MD)
    L.append("Every figure file the two ICLR drafts (`w17-paper/DRAFT-v2.tex`, `DRAFT-v2-style.tex`) include, one line per file. "
             "\"Where used\" comes from each draft's `\\figslot{stem}{caption}{label}` calls and its `figure` environments with "
             "`\\includegraphics` (the rendered Shapes3D figures); figure numbers, section numbers and pages are read from the compiled "
             "`.aux` (DRAFT-v2 pages; the style draft differs by at most %d page(s); figure numbers and labels are %s). "
             "\"Data source\" is from `PROVENANCE.json` (build figures; paths under `~/clu/evidence/outputs/`) or the owner's provenance "
             "(mirrored figures). \"Build hash\" is the SHA-256 in the installed two-build manifest, and %s. "
             "Generated by `pw1_draft_figures/make_index.py` (branch `pw1/draft-figures`, a port of `w22_paper_visuals/make_index.py`); "
             "the build figures were built by `w21_figures/build.sh` at commit `%s`. No number here is new evidence.\n"
             % (c["max_page_difference_between_drafts"], "identical" if c["placement_identical_in_both_drafts"] else "NOT identical",
                "it equals the installed file's own SHA-256 for all %d files" % c["files_here_png_pdf"] if not c["hash_problems"]
                else "⚠ PROBLEMS: " + "; ".join(c["hash_problems"]), c["build_commit"]))
    L.append("## Completeness, both ways\n")
    L.append("- **Draft → folder:** %d figure include targets in DRAFT-v2, %d in DRAFT-v2-style (same stems, labels and figure numbers). "
             "**%d / %d have both PDF and PNG here. Missing: %s.**" % (
                 c["include_targets_per_draft"]["DRAFT-v2.tex"], c["include_targets_per_draft"]["DRAFT-v2-style.tex"],
                 c["include_targets_union"] - len({m.rsplit(".", 1)[0] for m in c["missing_files"]}), c["include_targets_union"],
                 ", ".join(c["missing_files"]) or "none"))
    L.append("- **Folder → draft:** %d figure files (PNG+PDF) outside `retired-pw1/`; **%d placed, %d unreferenced** (%s). "
             "%d retired files sit in `retired-pw1/` and `rendered/retired-pw1/` (listed below; none is placed)." % (
                 c["files_here_png_pdf"], c["files_here_referenced"], len(c["files_here_unreferenced"]),
                 ", ".join(c["files_here_unreferenced"]) or "none", c["retired_files"]))
    L.append("- **PNG/PDF parity:** %d stems, %s.\n" % (c["stems_here"], "every one has both" if not c["png_pdf_parity_failures"]
                                                        else "MISSING A FORMAT: " + ", ".join(c["png_pdf_parity_failures"])))
    L.append("## Figures (one line per file)\n")
    L.append("| file | what it shows | where used | data source | build hash (SHA-256) |")
    L.append("|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["fig"], r["file"])):
        L.append("| `%s` | %s | %s | %s | `%s` (%s) |" % (r["file"], md(r["shows"]), md(r["placement"]), md(r["source"]),
                                                       r["manifest_sha256"], r["built_by"]))
    L.append("\n## Retired (moved into `retired-pw1/`, never deleted; not placed by either draft)\n")
    L.append("| file | what it was | SHA-256 |")
    L.append("|---|---|---|")
    for r in ret_rows:
        L.append("| `%s` | %s | `%s` |" % (r["file"], md(r["shows"]), r["sha256"]))
    L.append("\n## Other files\n")
    L.append("- `PROVENANCE.json`: per build figure, every source read with its SHA-256 (written by the build). "
             "`GLYPH-IDENTITY.json`: the task-figure / result-panel glyph identity check. `SHA256-w22-build{1,2}.txt`: the build's two-build manifest "
             "(the file name is the build script's).")
    L.append(readme_block.OTHER_CODE_MD)
    L.append("- `rendered/PROVENANCE-render.json`, `rendered/REQUESTS-pw1.json`: the rendered figures' provenance, drawn requests, the judge's "
             "six read-backs per panel, and the counts quoted in the captions.")
    L.append("- `FIGURES.md`: longer notes per figure; its PW-1 section is on top.")
    L.append("- Anything else in this folder is history from earlier builds (manifests and scripts of w17 to w22), kept, not used by the drafts.")
    open(os.path.join(PV, "README.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    rows, ret_rows, counts = main()
    write_readme(rows, ret_rows, counts)
    print(json.dumps(counts, indent=1))
    sys.exit(1 if counts["missing_files"] or counts["hash_problems"] or counts["png_pdf_parity_failures"] else 0)
