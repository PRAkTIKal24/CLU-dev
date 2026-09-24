"""w22 paper-visuals: build the README index of ~/clu/outputs/paper-visuals from artefacts.

Reads (never writes) the two drafts and their compiled .aux (from a scratch compile), the folder's
build manifests and PROVENANCE.json, and the owning spokes' manifests for the mirrored files.
Writes README.md into the folder and INDEX.json + COMPLETENESS.txt into the w22 report dir.

Every field except "what it shows" is derived here. "What it shows" is DESCRIPTIONS below: written
after opening each PNG (and, for build figures, the PROVENANCE notes), never from a filename.
"""
import hashlib
import json
import os
import re
import sys

PV = "/Users/batmac/clu/outputs/paper-visuals"
PAPER = "/Users/batmac/clu/outputs/w17-paper"
DRAFTS = ["DRAFT-v2.tex", "DRAFT-v2-style.tex"]
OUT = "/Users/batmac/clu/outputs/w22-paper-visuals"
AUX = os.path.join(OUT, "compile-evidence")
EV = "evidence/outputs"  # PROVENANCE sources are relative to /Users/batmac/clu/evidence/outputs

# Build manifests. The w22 build's own, and the owning spokes' for files mirrored in unchanged.
MANIFESTS = {
    "build": (os.path.join(PV, "SHA256-w22-build1.txt"), os.path.join(PV, "SHA256-w22-build2.txt"), ""),
    "task": ("/Users/batmac/clu/outputs/w19-task-figures/proof/SHA256-build1.txt",
             "/Users/batmac/clu/outputs/w19-task-figures/proof/SHA256-build2.txt", ""),
    "rendered": (os.path.join(PAPER, "figures/rendered/SHA256-w19-build1.txt"),
                 os.path.join(PAPER, "figures/rendered/SHA256-w19-build2.txt"), "rendered/"),
}

# Plain-words descriptions, one per figure stem (both formats share it). Written from the images.
DESCRIPTIONS = {
    "fig_task_shapes": "The image task, with no results: (a) the six things that vary between images (floor, wall and object colour, object size, shape, viewing angle); (b) a request (six values) turned into an image by the fixed decoder and read back by a separate classifier; (c, d) which value combinations are kept out of training.",
    "fig_task_cmazet": "The maze task, with no results: (a) one generated maze with coloured keys, doors and tinted floor, the agent and the goal; (b-d) three single moves (legal; blocked because the floor matches the key being held; legal because a matching door uses the key up first), each written as a five-word request and a four-fact answer.",
    "rendered/rc_w19_t7_per_arm_success_failure": "Image task, seed 0: for each of the five methods (CLU, MLP, LSTM, Transformer, continuous MLP), one held-out request it gets right and one it gets wrong, with every method's image for that request and the classifier's tick or cross.",
    "rendered/rc_w19_t7_rival_succeeds_clu_fails": "Image task, seed 0: one held-out request each on which the MLP, the LSTM or the Transformer is right and the CLU is wrong, every method's image side by side.",
    "rendered/rc_w19_t7_rival_succeeds_clu_fails_seed1": "Image task, seed 1: the only request, over all seeds, on which the continuous MLP is right and the CLU is wrong, every method's image side by side.",
    "fig_ladder": "Distinct valid new combinations per held-out corner as more factors are held out together (2 to 6), the CLU against the rival networks and a lookup-table floor, 3 seeds. One panel per rival training setup, and each panel title gives the rivals' training-set size and step count: 32,256 rows and 5,040 steps; 32,768 rows and 2,560 steps; 131,072 rows and 20,480 or 204,800 steps.",
    "fig_headline_steps": "Distinct valid new combinations per corner against the rivals' training steps (log scale), 3 seeds. (a) The MLP on 16,384, 32,768 and 131,072 training rows: at 20,480 steps it scores 38.6, 39.1 and 39.2 on the three pools, within about one of itself over an 8x range of data, while short runs sit near the lookup-table floor (7.7). So the rivals were short of training steps, not of data. (b) All four rival families on the same axis. The CLU line (about 48) is for reference.",
    "fig_crossover": "(a) Fraction of valid validation outputs as the number of stored clusters K grows from 5 to 32 (3 seeds, mean with a one-standard-deviation band), with the best of four sanity-check controls at zero; (b) the method's memory in bytes against K, split into its four parts and added up, with a 524,288-byte reference line.",
    "fig_deletion": "Deleting training data from the fitted method, 3 seeds, three deletions: one colour value, the red-and-sphere combination, and a random 1 % of items as a control. (a, b) The edited model matches a model refitted from scratch without the data, to within 1e-6. (c) Refusal behaviour, as text. (d, e) Output quality on the surviving corners before against after: red-and-sphere drops it by about 6.3, while the random-1 % control beside it moves it by at most 0.11. That control shows the drop comes from the deleted content, not from refit instability. (f) The edited method minus each rival retrained without the same rows.",
    "fig_encoder_adapt": "The CLU's margin over a 180k-parameter MLP baseline, per seed (3 seeds), for three encoder settings: frozen, with a small adapter, and with an anchor term. Lines join the same seed.",
    "fig_maze_readverb": "Earlier maze variant: fraction of valid routes per method (CLU, lookup table, three rival networks, three controls) under two ways of combining a request (averaging in latent space against merging plans), for unseen pairs (a) and multi-way combinations (b), with 3 seeds as dots and the pipeline's ceiling as a band.",
    "fig_maze_conditions": "Maze task, fraction of held-out multi-way transitions predicted exactly. (a) One row per input setup (seeds 0-2): the CLU against the best comparable rival, with the win, loss or comparable label. (b, c) Seeds 0-5 for the plain CLU (b) and for the CLU with colour-relational request coding (c): the best rival, the best rival given the same relational inputs, the top control, and a ceiling bar. Hard seeds 1 and 4 (by ceiling) are shaded. Every tie or loss is marked: in (b) a rival beats the CLU on seeds 1 and 5; in (c) a control ties the CLU on seed 4 (0.900) and a rival ties it on seed 5 (0.939).",
    "fig_maze_rel_tuples": "Maze task, CLU with colour-relational request coding, one heat-map per seed (0-5): the fraction of each held-out combination's rows predicted exactly, for the CLU, the best rival, the best rival given relational inputs, and the top control. On the right is the number of combinations on which each one exactly equals the CLU. All 16 of 16 are tied on seed 4 (control and relational rival) and on seed 5 (rival and relational rival).",
    "fig_cmazet_families": "Maze task, one held-out move per rule type (5 rows): the start state and move, the true next state, and each of 14 methods' predicted next state with a tick or cross.",
    "fig_cmazet_per_arm": "Maze task, seed 0: for every method, one held-out move whose next state it predicts exactly and one it gets wrong. Two methods have no exact prediction, so they show a failure only.",
    "fig_cmazet_door_then_tint": "Maze task, the hardest rule (a door opens, then the tinted-floor rule applies): one move per direction (north, south, west), every method's predicted next state. Only the 80,000-step LSTM gets one right.",
    "fig_cmazet_deletion": "Maze task after deleting colour 1: four surviving moves, showing the CLU before deletion, after the edit, and refitted without the data (all three agree), and the MLP as trained and retrained without it.",
    "fig_maze_requestlayer": "Audit of the maze request generator before and after a fix: (a) the number of keys on each optimal route; (b) success of scripted planners, for the old and the fixed maze variant. It contains no learned method.",
}

# Data source for mirrored files (no PROVENANCE.json entry in this folder): from the owners' records.
MIRROR_SOURCES = {
    "fig_task_shapes": "w19-task-figures/proof/PROVENANCE-task-figures.json (3dshapes.h5 at the runbook pin; value_holdout/{factor_index.npz,value_cells.json}; rendered/REQUESTS.json requests[0]; frozen decoder seed 0 01ef91f7…; classifier c8741ac4…)",
    "fig_task_cmazet": "w19-task-figures/proof/PROVENANCE-task-figures.json (CMAZE-T engine: w8 maze_colour + w9 dynamics.step + chlu/exp_w15_cmaze_t.py @9f80f59; seed-0 TRAIN-bank maze, generator seed 2000006)",
    "rendered/rc_w19_t7_per_arm_success_failure": "w17-paper/figures/rendered/{REQUESTS-w19,SAME-DECODER-w19}.json; evidence w12-single-design-crossover + w12b-nr6-and-kstar scored.json (seed 0, K*=32)",
    "rendered/rc_w19_t7_rival_succeeds_clu_fails": "w17-paper/figures/rendered/{REQUESTS-w19,SAME-DECODER-w19}.json (t7.counts_per_seed); evidence w12 / w12b T7 scored.json (seed 0)",
    "rendered/rc_w19_t7_rival_succeeds_clu_fails_seed1": "w17-paper/figures/rendered/{REQUESTS-w19,SAME-DECODER-w19}.json; evidence w12 / w12b T7 scored.json (seed 1, seed 1's decoder 001011bc…)",
}

# Why a file is present but unreferenced: read from the drafts' own comments.
UNREF_NOTE = {
    "fig_maze_requestlayer": "DRAFT-v2.tex:2166 / DRAFT-v2-style.tex:2152 comment: removed under REFEREE-v2 M5 (both request-layer environments on one figure)",
}

# The one pending slot (data/cmazet_panels.json status not_landed): not a file, recorded as pending.
PENDING = {
    "fig_cmazet_rel_per_arm": "CMAZE-T per-row panels for the CLU with colour-relational request coding. NOT DRAWN: gate 1 passed (192/192 objects hash-verified) but gate 2 is blocked because the persisted objects need Python module `w21_clu`, absent here (w22 figures-redraw2 NEEDS-ARTEFACTS.json). UNBLOCK: `w21_clu` is carried by the branch bundles now in ~/clu/ship/ (w21-rel-newseeds.bundle, w21-maze-t-clu-improve.bundle). No draft places it, so no draft is missing it.",
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


def toc_numbers(aux):
    """Numbered (sub)section entries from the compiled .aux, in order."""
    txt = open(aux).read()
    return [(lvl, num) for lvl, num in
            re.findall(r"\\contentsline \{(section|subsection)\}\{\\numberline \{([^}]*)\}", txt)]


def fig_numbers(aux):
    txt = open(aux).read()
    return {lab: (num, page) for lab, num, page in
            re.findall(r"\\newlabel\{(fig:[^}]*)\}\{\{([^}]*)\}\{([^}]*)\}", txt)}


def placements(draft):
    """figslot stem -> (label, line, section string), with section numbers taken from the .aux."""
    src = open(os.path.join(PAPER, draft)).read().splitlines()
    toc = toc_numbers(os.path.join(AUX, draft.replace(".tex", ".aux")))
    figs = fig_numbers(os.path.join(AUX, draft.replace(".tex", ".aux")))
    k = -1
    cur_sec = cur_sub = None
    out = {}
    refs = {}
    buf = None
    for i, raw in enumerate(src, 1):
        line = strip_comment(raw)
        for m in re.finditer(r"\\(section|subsection)(\*?)\{", line):
            lvl, star = m.group(1), m.group(2)
            title = line[m.end():].split("}")[0][:70]
            if star:
                if lvl == "section":
                    cur_sec, cur_sub = ("*", title), None
                continue
            k += 1
            assert toc[k][0] == lvl, (draft, i, toc[k], lvl)
            if lvl == "section":
                cur_sec, cur_sub = (toc[k][1], title), None
            else:
                cur_sub = (toc[k][1], title)
        for r in re.findall(r"\\(?:[cC]|auto)?ref\{(fig:[^}]*)\}", line):
            where = cur_sub or cur_sec
            refs.setdefault(r, [])
            tag = ("§%s %s" % (where[0], clean(where[1])[:40])) if where and where[0] != "*" else "appendix preamble"
            if tag not in refs[r]:
                refs[r].append(tag)
        if "\\figslot{" in line and "newcommand" not in line:
            buf = (i, cur_sec, cur_sub, "")
        if buf is not None:
            buf = (buf[0], buf[1], buf[2], buf[3] + line + "\n")
            m = re.match(r"\s*\\figslot\{([^}]*)\}(.*)", buf[3], re.S)
            lab = re.search(r"\}\{(fig:[^}]*)\}\s*$", buf[3].rstrip())
            if m and lab:
                where = buf[2] or buf[1]
                num, page = figs[lab.group(1)]
                out[m.group(1)] = dict(label=lab.group(1), line=buf[0], fig=num, page=page,
                                       section="§%s %s" % (where[0], clean(where[1])) if where[0] != "*" else clean(where[1]))
                buf = None
    assert k + 1 == len(toc), (draft, k + 1, len(toc))
    for v in out.values():
        v["cited_from"] = refs.get(v["label"], [])
    return out


def clean(t):
    for a, b in ((r"$\Kstar$", "K*"), (r"$n$", "n"), (r"$K$", "K"), (r"$K=5$", "K=5"), (r"$1\times$", "1x"),
                 ("Table~\\ref{tab:t7}", "Table T7")):
        t = t.replace(a, b)
    return t.strip()


def main():
    place = {d: placements(d) for d in DRAFTS}
    prov = json.load(open(os.path.join(PV, "PROVENANCE.json")))["figures"]
    man = {}
    for key, (m1, m2, pre) in MANIFESTS.items():
        a, b = read_manifest(m1, pre), read_manifest(m2, pre)
        assert a == b, ("manifest builds differ", key)
        for n, h in a.items():
            man[n] = (h, key)

    # every PNG/PDF in the folder
    files = []
    for root, _, fs in os.walk(PV):
        for f in fs:
            if f.endswith((".png", ".pdf")):
                files.append(os.path.relpath(os.path.join(root, f), PV))
    files.sort()
    stems = sorted({os.path.splitext(f)[0] for f in files})

    lines, rows, problems = [], [], []
    referenced = set()
    for d in DRAFTS:
        referenced |= set(place[d])
    # (1) every include target in either draft has PNG and PDF here
    missing = []
    for stem in sorted(referenced):
        for ext in (".pdf", ".png"):
            if not os.path.exists(os.path.join(PV, stem + ext)):
                missing.append(stem + ext)
    # (2) every file here is referenced or UNREFERENCED
    unref = [f for f in files if os.path.splitext(f)[0] not in referenced]
    # parity
    parity = [s for s in stems if not (os.path.exists(os.path.join(PV, s + ".pdf")) and
                                       os.path.exists(os.path.join(PV, s + ".png")))]
    same_place = all(place[DRAFTS[0]][s]["label"] == place[DRAFTS[1]][s]["label"] and
                     place[DRAFTS[0]][s]["fig"] == place[DRAFTS[1]][s]["fig"]
                     for s in referenced if s in place[DRAFTS[0]] and s in place[DRAFTS[1]])
    only_one = sorted(set(place[DRAFTS[0]]) ^ set(place[DRAFTS[1]]))
    page_diff = max(abs(int(place[DRAFTS[0]][x]["page"]) - int(place[DRAFTS[1]][x]["page"])) for x in referenced)

    for f in files:
        stem = os.path.splitext(f)[0]
        installed = sha(os.path.join(PV, f))
        mh, mkey = man.get(f, (None, None))
        if mh is None:
            problems.append("NO MANIFEST HASH: " + f)
        elif mh != installed:
            problems.append("HASH MISMATCH: %s installed %s manifest %s" % (f, installed, mh))
        if stem in prov:
            src = "; ".join(prov[stem]["sources"]) if isinstance(prov[stem]["sources"], list) else str(prov[stem]["sources"])
            if len(prov[stem]["sources"]) > 4:
                roots = sorted({"/".join(s.split("/")[:2]) if s.split("/")[1] not in ("runs", "rivals", "test", "val", "replay", "t1", "artefacts")
                                else s.split("/")[0] + "/" + s.split("/")[1] + "/…" for s in prov[stem]["sources"]})
                src = "%d files under: %s (full list: PROVENANCE.json figures.%s.sources)" % (
                    len(prov[stem]["sources"]), "; ".join(roots), stem)
            src = "evidence/outputs/ " + src
        else:
            src = MIRROR_SOURCES.get(stem)
            if src is None:
                problems.append("NO DATA SOURCE: " + f)
                src = "?"
        if stem in referenced:
            p = place[DRAFTS[0]].get(stem) or place[DRAFTS[1]].get(stem)
            where = "Figure %s, `%s`, %s (p. %s)" % (p["fig"], p["label"], p["section"], p["page"])
            pls = [d.replace(".tex", "") + ":%d" % place[d][stem]["line"] for d in DRAFTS if stem in place[d]]
            where += "; slot at " + ", ".join(pls)
            where += "; cited in text from: " + (", ".join(p["cited_from"]) if p["cited_from"] else "nowhere (placed, never \\ref'd)")
        else:
            where = "**UNREFERENCED** by both drafts. " + UNREF_NOTE.get(stem, "")
        desc = DESCRIPTIONS.get(stem)
        if desc is None:
            problems.append("NO DESCRIPTION: " + f)
            desc = "?"
        builder = {"build": "this folder's build (w22)", "task": "w19 task-figures (mirrored)",
                   "rendered": "w19 success-failure-panels (mirrored)"}.get(mkey, "?")
        rows.append(dict(file=f, shows=desc, placement=where, source=src, sha256=installed,
                         manifest_sha256=mh, built_by=builder,
                         referenced=stem in referenced))

    counts = dict(
        drafts=DRAFTS,
        include_targets_per_draft={d: len(place[d]) for d in DRAFTS},
        include_targets_union=len(referenced),
        include_targets_with_pdf_and_png_here=len(referenced) - len({m.rsplit(".", 1)[0] for m in missing}),
        missing_files=missing,
        files_here_png_pdf=len(files),
        files_here_referenced=len(files) - len(unref),
        files_here_unreferenced=unref,
        stems_here=len(stems),
        png_pdf_parity_failures=parity,
        placement_identical_in_both_drafts=same_place,
        max_page_difference_between_drafts=page_diff,
        stems_placed_in_only_one_draft=only_one,
        hash_problems=problems,
        pending_slots=PENDING,
    )
    json.dump(dict(counts=counts, rows=rows), open(os.path.join(OUT, "INDEX.json"), "w"), indent=1)
    return rows, counts


SUPPORT = [
    ("PROVENANCE.json", "Per build figure: every source artefact read and the SHA-256 of each source; build notes. Written by the build (hash in SHA256-w22-build1.txt)."),
    ("FIGURES.md", "Longer notes per figure from the three build passes (w17, w21, w22), colour roles, pending slots. History sections are marked superseded."),
    ("SHA256-w22-build1.txt, SHA256-w22-build2.txt", "The build's manifest: SHA-256 of its 26 figure files + PROVENANCE.json from two independent builds (identical). This is the \"last-build hash\" column for every build figure."),
    ("SHA256-mirrored-w19-task-figures-build{1,2}.txt", "Copied here from w19-task-figures/proof/SHA256-build{1,2}.txt: the manifest for fig_task_* (built by w19 task-figures, not by this folder's build)."),
    ("rendered/SHA256-w19-build{1,2}.txt", "Copied here from w17-paper/figures/rendered/: the manifest for rendered/rc_w19_* (built by w19 success-failure-panels). It also lists rc_w19_cmazet_* files, which are not in this folder and not used by either draft."),
    ("build.sh", "Builds all 26 build figures twice, checks the two builds are byte-identical, installs into w17-paper/figures/ and here. ⚠ NOT runnable from this folder (see Rebuild below): run the identical copy at w21_figures/build.sh in a CHLU worktree."),
    ("make_figures.py, make_figures_w21.py, make_figures_w22.py, maze_six.py, cmazet_panels.py", "The build code (w17 figures; w21 redraw library; w22 figures; six-seed maze reader; maze panel renderer)."),
    ("validate_palette.py", "Colour-palette contrast / colour-blind check used when choosing the figure colours."),
    ("data/ladder_rival_sets.json, data/maze_conditions.json, data/cmazet_panels.json", "What the build draws, as data: ladder rival sets, maze condition rows, maze panel sets (one set, fig_cmazet_rel_per_arm, is not_landed)."),
    ("data/cmazet_items.json, data/cmazet_items.REPRO.json", "Per-item re-read of the persisted maze objects drawn by fig_cmazet_*, and its reproduction check against the scored numbers."),
    ("README.md", "This index (generated by w22_paper_visuals/make_index.py on branch w22/paper-visuals)."),
]


def md(t):
    return t.replace("|", "\\|").replace("\n", " ")


def write_readme(rows, counts):
    rb = json.load(open(os.path.join(OUT, "REBUILD.json")))
    a2 = rb["attempt2"]
    L = []
    L.append("# paper-visuals — index\n")
    intro = ("Every figure file the two ICLR drafts (`w17-paper/DRAFT-v2.tex`, `DRAFT-v2-style.tex`) include, "
             "one line per file. The \"where used\" column comes from each draft's `\\figslot{stem}{caption}{label}` call "
             "(the macro that does the `\\includegraphics`), with figure, section and page numbers read from a compiled "
             "`.aux` (DRAFT-v2 page numbers; the style draft differs by at most %d pages; figure numbers and labels are %s). "
             "\"Data source\" is from `PROVENANCE.json` (build figures) or the owning spoke's provenance (mirrored figures); "
             "paths are relative to `~/clu/evidence/outputs/` unless they start with `w17-paper/` or `w19-`, which are under `~/clu/outputs/`. "
             "\"Build hash\" is the SHA-256 recorded in the installed build manifest, and %s. "
             "Generated by `w22_paper_visuals/make_index.py` (branch `w22/paper-visuals`). No number here is new evidence.\n")
    L.append(intro % (counts["max_page_difference_between_drafts"],
                      "identical" if counts["placement_identical_in_both_drafts"] else "NOT identical",
                      "it equals the installed file's own SHA-256 for all %d files" % counts["files_here_png_pdf"]
                      if not counts["hash_problems"] else "⚠ HASH PROBLEMS: " + "; ".join(counts["hash_problems"])))
    L.append("## Completeness, both ways\n")
    L.append("- **Draft → folder:** %d figure include targets in DRAFT-v2, %d in DRAFT-v2-style (same %d stems, same labels, same figure numbers). "
             "**%d / %d have both PDF and PNG here. Missing: %s.**" % (
                 counts["include_targets_per_draft"]["DRAFT-v2.tex"], counts["include_targets_per_draft"]["DRAFT-v2-style.tex"],
                 counts["include_targets_union"], counts["include_targets_with_pdf_and_png_here"], counts["include_targets_union"],
                 ", ".join(counts["missing_files"]) or "none"))
    L.append("- **Folder → draft:** %d figure files (PNG+PDF) here; **%d referenced, %d UNREFERENCED** (%s)." % (
        counts["files_here_png_pdf"], counts["files_here_referenced"], len(counts["files_here_unreferenced"]),
        ", ".join(counts["files_here_unreferenced"]) or "none"))
    L.append("- **PNG/PDF parity:** %d stems, %s." % (counts["stems_here"], "every one has both" if not counts["png_pdf_parity_failures"]
                                                     else "MISSING A FORMAT: " + ", ".join(counts["png_pdf_parity_failures"])))
    L.append("- **This folder alone is enough:** in a scratch copy of `w17-paper/` with `figures/` replaced by this folder, both drafts compile "
             "(pdflatex exit 0, 0 errors) and include 17 / 17 figures, with 0 \"Figure pending\" fallbacks.")
    L.append("- **Pending slot (not a file, not placed by any draft):** `fig_cmazet_rel_per_arm`. " + PENDING["fig_cmazet_rel_per_arm"] + "\n")
    L.append("## Rebuild check (%s)\n" % "byte-identical" if a2["identical_to_installed_and_manifest"] == a2["files"] else "## Rebuild check: DIFFERS\n")
    L.append("- `w21_figures/build.sh` (identical to this folder's `build.sh` and code) was run from a CHLU worktree into scratch destinations. "
             "Two builds matched, and **%d / %d** of its outputs (26 figure files + PROVENANCE.json) are byte-identical to the files installed here "
             "and to `SHA256-w22-build1.txt`. Nothing was overwritten." % (a2["identical_to_installed_and_manifest"], a2["files"]))
    L.append("- ⚠ **`bash build.sh` run in place in this folder fails** before drawing anything: `cmazet_panels.py` takes the repository to be "
             "the parent of its own folder (`~/clu/outputs`, which has no `chlu`) and asserts on it. The code is right, but the copy here is "
             "not self-contained. To rebuild: `cd <a CHLU worktree at 102d1de or later>; source ~/clu/env_bm.sh; bash w21_figures/build.sh <dest1> <dest2>`.")
    L.append("- The 10 mirrored files (`fig_task_*`, `rendered/rc_w19_t7_*`) are not built by this build. They match their owners' two-build manifests "
             "(copied here, see below) but were **not rebuilt**: the rendered panels need the frozen image decoder run, and the task figures need the dataset file.\n")
    L.append("## Figures (one line per file)\n")
    L.append("| file | what it shows | where used | data source | build hash (SHA-256) |")
    L.append("|---|---|---|---|---|")
    order = sorted(rows, key=lambda r: (not r["referenced"], int(re.search(r"Figure (\d+)", r["placement"]).group(1))
                                        if r["referenced"] else 99, r["file"]))
    for r in order:
        L.append("| `%s` | %s | %s | %s | `%s` (%s) |" % (r["file"], md(r["shows"]), md(r["placement"]), md(r["source"]),
                                                       r["manifest_sha256"], r["built_by"]))
    L.append("\nThe PDF is what the drafts include. For the 26 build files, the PNG comes from the same `fig.savefig` sequence in `save()` (make_figures.py / make_figures_w21.py) as the PDF, so it is not converted from the PDF. For the 10 mirrored files, both formats come from their owners' builds.\n")
    L.append("## Other files in this folder\n")
    L.append("| file | what it is |")
    L.append("|---|---|")
    for f, d in SUPPORT:
        L.append("| `%s` | %s |" % (f, md(d)))
    L.append("\nThe report for this index, with its log, counts, rebuild evidence and findings, is in `~/clu/outputs/w22-paper-visuals/`: `FINDINGS.md`, `INDEX.json`, `REBUILD.json` and `LOG.md`.")
    open(os.path.join(PV, "README.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    rows, counts = main()
    write_readme(rows, counts)
    print(json.dumps(counts, indent=1))
    for r in rows:
        print(r["file"], "|", r["placement"][:70], "|", r["sha256"][:12], r["manifest_sha256"] and r["manifest_sha256"][:12])
    sys.exit(1 if counts["missing_files"] or counts["hash_problems"] or counts["png_pdf_parity_failures"] else 0)
