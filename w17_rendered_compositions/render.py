"""w17 rendered compositions -- the driver.

    python -m w17_rendered_compositions.render --placeholder   # what runs on batmac today
    python -m w17_rendered_compositions.render --render        # once NEEDS-ARTEFACTS.json is staged

`--render` expands the corners `selection` fixed into exact request tuples, reads every
arm on those requests through the scorer's own `exp_gen_read.arm_features`, renders every
panel through `OneDecoder` (ONE hash-asserted decoder for all arms), marks each panel with
the scorer's own verdict (`judge_valid & accept_rule`), re-derives each rendered corner's
`n_valid` and REFUSES if it differs from the scored.json it was selected from, and writes
REQUESTS.json with the exact tuples.

`--placeholder` writes the same four figures as DECLARED placeholders -- hatched panels,
verdict marks only where the selection rule fixes them from the scored counts, and a red
banner -- plus a corner-level REQUESTS.json.  It never draws a picture of anything.
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from . import selection as S
from .figure import draw_grid
from .same_decoder import OneDecoder, assert_same_decoder

REPO = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("W17_RC_OUT", "/Users/batmac/clu/outputs/w17-paper/figures/rendered"))
SPOKE_OUT = Path("/Users/batmac/clu/outputs/w17-rendered-compositions")
_STAGE_ROOT = Path(os.environ.get("W17_RC_STAGE", "/Users/batmac/clu/stage"))
#: the ThinkPad's C:/chlu/outputs tree, as the Hub mirrored it (<W17_RC_STAGE>/outputs/)
STAGE = _STAGE_ROOT / "outputs" if (_STAGE_ROOT / "outputs").is_dir() else _STAGE_ROOT
#: the comparators' .rival.eqx were staged WITHOUT their .rival.json manifests (those are in
#: the evidence base); load_rival wants both side by side, so they are joined here by
#: symlink -- nothing is copied or rewritten.
ASSEMBLED = SPOKE_OUT / "_assembled_comparators"

FIG = {
    "set1": "rc_set1_clu_composes_rivals_fail",
    "set2": "rc_set2_all_arms_succeed",
    "set3": "rc_set3_all_arms_fail",
    "deletion": "rc_deletion_red_sphere",
}
FACTOR_SHORT = {"floor_hue": "floor", "wall_hue": "wall", "object_hue": "object hue",
                "scale": "scale", "shape": "shape", "orientation": "orient."}
SHAPE_NAMES = {0: "cube", 1: "cylinder", 2: "sphere", 3: "oblong"}   # w12_deletion_rows.py:77
DELETION_ARMS = (("mlp", "MLP", "#1baf7a"), ("lstm", "LSTM", "#eda100"),
                 ("transformer", "Transformer", "#e87ba4"),
                 ("cont_mlp", "Continuous\nMLP", "#008300"))
CONJ = "conj-object_hue0-shape2"     # the comparators' red-sphere pin (BASELINES-READY pins)


def _now():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg):
    line = f"{_now()} {msg}"
    print(line, flush=True)
    with open(SPOKE_OUT / "LOG.md", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# ---------------------------------------------------------------------------------------
# placeholder mode
# ---------------------------------------------------------------------------------------
BANNER = ("DECLARED PLACEHOLDER -- NOT RENDERED\nfrozen decoder, CLU arm and rival "
          "weights not staged (NEEDS-ARTEFACTS.json)")


def placeholder() -> dict:
    sel = S.select()
    labels = [l for _, l, _ in S.ARMS]
    colours = [c for _, _, c in S.ARMS]
    k = S.REQUESTS_PER_CORNER
    out = {}

    def rows_for(corners):
        return [f"{cid}\nrequest {i + 1} of the\nselection rule" for cid in corners
                for i in range(k)]

    s1 = sel["set1_clu_composes_rivals_fail"]["corners"]
    r1 = rows_for(s1)
    v1 = [[a == S.CLU for a, _, _ in S.ARMS] for _ in r1]
    out["set1"] = draw_grid(OUT / FIG["set1"], labels, colours, r1,
                            [[None] * len(labels) for _ in r1], v1,
                            title="Set 1 -- the CLU composes, every rival fails",
                            placeholder_banner=BANNER)

    s2 = sel["set2_all_arms_succeed"]["candidate_corners_per_seed"][str(S.RENDER_SEED)]
    r2 = [f"{cid}\ncandidate -- may be EMPTY\n(then DECLARED)" for cid in s2] * 2
    out["set2"] = draw_grid(OUT / FIG["set2"], labels, colours, r2,
                            [[None] * len(labels) for _ in r2],
                            [[None] * len(labels) for _ in r2],
                            title="Set 2 -- every arm succeeds",
                            placeholder_banner=BANNER)

    s3 = sel["set3_all_arms_fail"]["corners"]
    r3 = rows_for(s3)
    v3 = [[False] * len(labels) for _ in r3]
    out["set3"] = draw_grid(OUT / FIG["set3"], labels, colours, r3,
                            [[None] * len(labels) for _ in r3], v3,
                            title="Set 3 -- every arm fails",
                            placeholder_banner=BANNER)

    dl = ["CLU"] + [l for _, l, _ in DELETION_ARMS]
    dc = ["#2a78d6"] + [c for _, _, c in DELETION_ARMS]
    out["deletion"] = draw_grid(
        OUT / FIG["deletion"], dl, dc,
        ["before deletion\n(CLU; rival as trained)",
         "after deletion\n(CLU downdate;\nrival retrained without)"],
        [[None] * len(dl) for _ in range(2)], [[None] * len(dl) for _ in range(2)],
        title="Red sphere, before and after deleting every red-sphere fit row",
        placeholder_banner=BANNER)

    req = {
        "schema": "w17-rendered-compositions/requests/1",
        "STATUS": "PLACEHOLDER -- CORNER-LEVEL ONLY",
        "why": ("the exact 6-factor tuples are the corners' dataset rows, expanded by "
                "exp_gen_factor_wells._cell_mask_rows from value_holdout/value_cells.json + "
                "factor_index.npz, neither of which is on this machine (NEEDS-ARTEFACTS.json). "
                "`--render` overwrites this file with the exact tuples it rendered."),
        "written_utc": _now(),
        "selection": sel,
        "deletion_request_rule": {
            "scope": "conjunction_live -- every fit row carrying a RED SPHERE (object_hue "
                     "index 0 AND shape index 2); the task file's red-cube pin is vacuous "
                     "on this split (deletion.json scopes.conjunction.vacuous_reason)",
            "pick": "among the deleted fit rows: largest scale index, then largest "
                    "orientation index, then smallest row id (the task's 'red sphere, "
                    "large, tilted')",
            "rival_before": "<family>__as_trained",
            "rival_after": f"<family>__retrain_without__{CONJ}",
            "clu_after": "the store downdated by the deleted rows' features, atoms "
                         "re-eigendecomposed, address book carried "
                         "(scripts/w12_deletion_rows.py, verbatim path)",
            "floor_excluded": "the lookup table has no retrain-without comparator in "
                              "w12-deletion-baselines, so it has no after-panel; it is left "
                              "out of the deletion panel rather than shown half",
        },
    }
    (OUT / "REQUESTS.json").write_text(json.dumps(req, indent=2), encoding="utf-8")
    status = {
        "STATUS": "PLACEHOLDER",
        "written_utc": _now(),
        "what_is_real": ["the corner selection (from the scored per-corner verdicts)",
                         "the verdict marks on sets 1 and 3, which the selection rule fixes "
                         "from those counts", "the figure layout, arm names and colours",
                         "the same-decoder guard (tests/test_w17_rendered_same_decoder.py)"],
        "what_is_not": ["every image", "set 2's existence at request level",
                        "every deletion-panel verdict", "the exact request tuples"],
        "to_finish": "stage NEEDS-ARTEFACTS.json, then `python -m "
                     "w17_rendered_compositions.render --render`",
        "files": out,
    }
    (OUT / "STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    shutil.copy(SPOKE_OUT / "NEEDS-ARTEFACTS.json", OUT / "NEEDS-ARTEFACTS.json")
    return status


# ---------------------------------------------------------------------------------------
# render mode
# ---------------------------------------------------------------------------------------
def preflight(seed: int = S.RENDER_SEED) -> list:
    """Every file `--render` opens at the rendered seed, and whether it is staged."""
    data = Path(os.environ.get("CHLU_GEN_DATA_ROOT", STAGE / "_data")) / "shapes3d" / "value_holdout"
    need = [
        Path(os.environ.get("CHLU_GEN_DECLIFT_DIR", STAGE / "gen-decoder-lift" / "winner"))
        / f"decoder_v1_s{seed}.pkl",
        Path(os.environ.get("CHLU_GEN_DECLIFT_DIR", STAGE / "gen-decoder-lift" / "winner"))
        / "winner_hashes.json",
        STAGE / "gen-external-classifier" / f"seed_{seed}" / "external_classifier.pkl",
        STAGE / "gen-external-classifier" / f"seed_{seed}" / "classifier_manifest.json",
        data / "factor_index.npz", data / "value_cells.json",
        STAGE / S.SCORED[seed].replace("scored.json", "arm.pkl"),
        STAGE / "spoke-a-rivals" / "rivals" / f"seed_{seed}" / "table.rival.pkl",
        STAGE / "hub-factor-wells" / "run" / "partitions.npz",
        STAGE / "spoke-a-rivals" / "cache_generation2" / "shapes3d_recon" / f"seed_{seed}"
        / "features.npy",
    ]
    croot = STAGE / "w12-deletion-baselines" / "comparators" / f"seed_{seed}"
    eroot = S.EVIDENCE / "w12-deletion-baselines" / "comparators" / f"seed_{seed}"
    for f, _, _ in DELETION_ARMS:
        for v in ("as_trained", f"retrain_without__{CONJ}"):
            need += [eroot / f"{f}__{v}.rival.json", croot / f"{f}__{v}.rival.eqx"]
    return [(str(p), p.exists()) for p in need]


def assemble_comparators(seed: int) -> Path:
    """<ASSEMBLED>/seed_N/{name}.rival.json -> evidence, {name}.rival.eqx -> stage."""
    d = ASSEMBLED / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    croot = STAGE / "w12-deletion-baselines" / "comparators" / f"seed_{seed}"
    eroot = S.EVIDENCE / "w12-deletion-baselines" / "comparators" / f"seed_{seed}"
    for eqx in sorted(croot.glob("*.rival.eqx")):
        js = eroot / eqx.name.replace(".rival.eqx", ".rival.json")
        for src in (eqx, js):
            dst = d / src.name
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src)
    return ASSEMBLED


def _setenv(skip_deletion: bool = False):
    miss = [p for p, ok in preflight() if not ok]
    if skip_deletion:
        miss = [p for p in miss if not p.endswith("features.npy")]
    if miss:
        log(f"--render REFUSED: {len(miss)} staged file(s) missing under W17_RC_STAGE={STAGE}: "
            + "; ".join(miss))
        raise SystemExit(f"{len(miss)} artefact(s) not staged -- see NEEDS-ARTEFACTS.json:\n  "
                         + "\n  ".join(miss))
    os.environ.setdefault("CHLU_GEN_DECLIFT_DIR", str(STAGE / "gen-decoder-lift" / "winner"))
    # classifier_dir() = BRIDGE_ROOT/../../gen-external-classifier/seed_N
    os.environ.setdefault("CHLU_GEN_BRIDGE_ROOT", str(STAGE / "_bridge" / "bridge"))
    os.environ.setdefault("CHLU_GEN_FEATURES_ROOT",
                          str(STAGE / "spoke-a-rivals" / "cache_generation2"))
    if "CHLU_GEN_DATA_ROOT" not in os.environ:
        raise SystemExit("CHLU_GEN_DATA_ROOT is not set: it must name the dir holding "
                         "shapes3d/value_holdout/{factor_index.npz,value_cells.json}")
    sys.path.insert(0, str(REPO / "scripts"))


def _label(grid, vals_row) -> str:
    parts = []
    for f, name in enumerate(grid.names):
        v = float(vals_row[f])
        if name == "shape":
            parts.append(SHAPE_NAMES[int(round(v))])
        else:
            parts.append(f"{FACTOR_SHORT.get(name, name)} {v:.3g}")
    return "\n".join([", ".join(parts[i:i + 2]) for i in range(0, len(parts), 2)])


def _load_arms(seed, grid, pool_vals):
    import pickle
    from chlu.eval.rivals import w4_api
    from chlu.eval.rivals.w12_deletion_baselines import load_comparator

    armp = STAGE / S.SCORED[seed].replace("scored.json", "arm.pkl")
    with open(armp, "rb") as fh:
        saved = pickle.load(fh)
    if int(saved["K"]) != 32 or int(saved["seed"]) != int(seed):
        raise SystemExit(f"{armp} holds K={saved['K']} seed={saved['seed']}, not K*=32 seed={seed}")
    objs = {S.CLU: {"arm": S.CLU, "read": "tensor", "kind": "span", "store": saved["store"],
                    "basis": saved["basis"], "address": saved["address"], "protos": None,
                    "head_bytes": {"head_bytes": 0}}}
    croot = assemble_comparators(seed)
    for a in S.OTHERS:
        rv = (w4_api.load_rival(seed, "table", root=STAGE / "spoke-a-rivals" / "rivals")
              if a == "table" else load_comparator(seed, a, root=croot))
        objs[a] = {"arm": a, "read": "rival", "rival": rv, "rival_partial": None,
                   "store": None, "head_bytes": {"head_bytes": 0}}
    return objs, saved


def _read_corner(objs, od, c, grid, bitmap):
    from chlu.experiments import exp_gen_factor_wells as fw
    from chlu.experiments import exp_gen_read as R
    vals = fw._request_values_from_rows(grid, c["rows"])
    res = {}
    for a, obj in objs.items():
        feats, oor = R.arm_features(obj, vals, grid)
        acc = fw.accept_rule(vals, oor, np.zeros(vals.shape[0], bool), bitmap, grid)
        imgs, stamps = od.render(feats, arm=a)
        valid, tup = od.verdict(vals, acc, imgs, grid)
        res[a] = {"imgs": imgs, "stamps": stamps, "valid": valid, "tuples": tup,
                  "accept": acc}
    return vals, res


def _reproduce(res, cid, seed):
    """The rendered corner must reproduce the scored n_valid for EVERY arm, or we stop."""
    pc = S.per_corner(seed)
    got, want = {}, {}
    for a in res:
        got[a] = int(res[a]["valid"].sum())
        want[a] = int(pc[a][cid]["n_valid"])
    if got != want:
        raise SystemExit(f"REPRODUCTION FAILED on {cid} seed {seed}: rendered n_valid {got} "
                         f"!= scored {want} ({S.SCORED[seed]}). The figure would not show "
                         "what was scored; nothing is written.")
    return {"corner_id": cid, "seed": seed, "n_valid_rendered": got,
            "n_valid_scored": want, "source": str(S.EVIDENCE / S.SCORED[seed]),
            "reproduced": True}


def _pick(vals, res, pattern, k):
    order = [a for a, _, _ in S.ARMS]
    V = np.stack([res[a]["valid"] for a in order], 1)
    want = np.array([pattern(a) for a in order], bool)
    hit = np.flatnonzero((V == want[None, :]).all(1))
    return hit[:k]


def render(scan_nway_all_fail: bool = True, skip_deletion: bool = False) -> dict:
    _setenv(skip_deletion)
    from chlu.experiments import exp_gen_factor_wells as fw

    seed = S.RENDER_SEED
    sel = S.select(seed)
    fw.PINS["condition"] = "test"
    fw.PINS["seed"] = seed
    cond = fw.load_condition()
    grid = cond["grid"]
    bitmap = fw.value_presence_bitmap(grid, cond["train_rows"])
    corners = {c["corner_id"]: c for c in fw.pair_subcorners(cond) + fw.nway_corners(cond)}
    judge = fw.Judge(seed, log=log)
    winner = S.EVIDENCE / S.WINNER_HASHES
    od = OneDecoder.from_judge(judge, seed, winner)
    objs, saved = _load_arms(seed, grid, None)
    log(f"render: seed {seed} decoder {od.hash[:16]} classifier {od.classifier_hash[:16]}")

    # positive control: this render path == the scorer's Judge.read_back
    from chlu.experiments import exp_gen_read as R
    c0 = corners[sel["set1_clu_composes_rivals_fail"]["corners"][0]]
    v0 = fw._request_values_from_rows(grid, c0["rows"])
    f0, _ = R.arm_features(objs[S.CLU], v0, grid)
    pos = od.positive_control(judge, f0)

    stamps, repro, requests, figs = [], [], [], {}
    labels = [l for _, l, _ in S.ARMS]
    colours = [c for _, _, c in S.ARMS]
    order = [a for a, _, _ in S.ARMS]
    k = S.REQUESTS_PER_CORNER

    def emit(set_key, rows):
        """rows: list of (corner_id, vals_row, row_id, res, i, provenance)"""
        imgs, verd, rl = [], [], []
        for cid, vrow, rid, res, i, prov in rows:
            imgs.append([res[a]["imgs"][i] for a in order])
            verd.append([bool(res[a]["valid"][i]) for a in order])
            fam = {"pair": "held-out 2-way corner", "nway": "held-out 6-factor box"}.get(
                corners[cid]["family"] if cid in corners else "", prov)
            rl.append(fam + "\n" + _label(grid, vrow))
            for a in order:
                stamps.append(res[a]["stamps"][i])
            requests.append({
                "set": set_key, "corner_id": cid, "provenance": prov, "dataset_row": int(rid),
                "factor_names": list(grid.names),
                "factor_values": [float(x) for x in vrow],
                "verdict": {a: bool(res[a]["valid"][i]) for a in order},
                "accepted": {a: bool(res[a]["accept"][i]) for a in order},
                "judged_tuple": {a: [int(x) for x in res[a]["tuples"][i]] for a in order},
                "seed": seed})
        return imgs, verd, rl

    # set 1 and set 3 ------------------------------------------------------------------
    patterns = {"set1": lambda a: a == S.CLU, "set3": lambda a: False,
                "set2": lambda a: True}
    titles = {"set1": "Set 1 -- the CLU composes, every rival fails",
              "set2": "Set 2 -- every arm succeeds",
              "set3": "Set 3 -- every arm fails"}
    cache = {}

    def corner(cid):
        if cid not in cache:
            vals, res = _read_corner(objs, od, corners[cid], grid, bitmap)
            repro.append(_reproduce(res, cid, seed))
            cache[cid] = (vals, res)
        return cache[cid]

    declared = {}
    for sk, key in (("set1", "set1_clu_composes_rivals_fail"),
                    ("set3", "set3_all_arms_fail")):
        rows = []
        for cid in sel[key]["corners"]:
            vals, res = corner(cid)
            for i in _pick(vals, res, patterns[sk], k):
                rows.append((cid, vals[i], corners[cid]["rows"][i], res, i, "held-out test corner"))
        if sk == "set3" and scan_nway_all_fail and not any(
                corners[c]["family"] == "nway" for c in sel[key]["corners"]):
            # no n-way corner qualifies on the COUNTS; look for an all-fail n-way REQUEST
            for cid in sorted(c for c in corners if corners[c]["family"] == "nway"):
                vals, res = corner(cid)
                hit = _pick(vals, res, patterns["set3"], 1)
                if hit.size:
                    rows.append((cid, vals[hit[0]], corners[cid]["rows"][hit[0]], res,
                                 hit[0], "held-out test corner (n-way scan)"))
                    break
            else:
                declared["set3_nway"] = ("no held-out n-way request fails for every arm at "
                                         "seed 0; set 3 is 2-way only")
        imgs, verd, rl = emit(sk, rows)
        figs[sk] = draw_grid(OUT / FIG[sk], labels, colours, rl, imgs, verd, title=titles[sk])

    # set 2 ------------------------------------------------------------------------------
    # Every held-out corner is scanned (the counts' candidate corners are necessary, not
    # sufficient).  MEASURED (diag/covalid_scan.json): no held-out request at seed 0, 1 or
    # 2 is valid for all six columns, and every 5-of-6 request fails only the continuous
    # MLP.  So the all-six set is DECLARED, and the figure shows the nearest pattern with
    # the continuous MLP's cross left in place -- never a SEEN row passed off as held-out.
    order_c = sorted(corners, key=lambda c: (corners[c]["family"] != "pair", c))
    six = []
    for cid in order_c:
        vals, res = corner(cid)
        six += [(cid, i) for i in _pick(vals, res, patterns["set2"], 1 << 30)]
    set2_pattern = "all six columns valid"
    if not six:
        n_req = sum(len(corners[c]["rows"]) for c in order_c)
        declared["set2_all_six"] = (
            f"DECLARED NOT BUILDABLE: 0 of {n_req} held-out requests at seed {seed} are "
            "valid for every column (all six arms). Shown instead: the nearest pattern, "
            "every arm EXCEPT the continuous MLP valid, with its cross shown. Seeds 1 and 2 "
            "were scanned too (diag/covalid_scan.json): 0 all-six there either.")
        patterns["set2"] = lambda a: a != "cont_mlp__as_trained"
        set2_pattern = "every column valid EXCEPT the continuous MLP (all-six set is empty)"
        titles["set2"] = ("Set 2 -- every arm succeeds except the continuous MLP\n"
                          "(no held-out request succeeds for all six)")
    rows2 = []
    for cid in order_c:
        if len(rows2) >= 2 * k:
            break
        vals, res = corner(cid)
        hit = _pick(vals, res, patterns["set2"], 1)
        if hit.size:
            i = int(hit[0])
            rows2.append((cid, vals[i], corners[cid]["rows"][i], res, i, "held-out test corner"))
    declared["set2_pattern_shown"] = set2_pattern
    if rows2:
        imgs, verd, rl = emit("set2", rows2)
        figs["set2"] = draw_grid(OUT / FIG["set2"], labels, colours, rl, imgs, verd,
                                 title=titles["set2"])
    else:
        declared["set2"] = "DECLARED NOT BUILDABLE: no held-out request meets either pattern"

    # deletion ---------------------------------------------------------------------------
    if skip_deletion:
        del_info = {"STATUS": "NOT RENDERED IN THIS PASS (--skip-deletion): features.npy "
                              "not yet staged"}
        declared["deletion"] = del_info["STATUS"]
    else:
        figs["deletion"], del_info = deletion_panel(objs, saved, od, grid, cond, bitmap,
                                                    stamps, requests)

    same = assert_same_decoder(stamps, winner, seed)
    same["positive_control"] = pos
    (OUT / "SAME-DECODER.json").write_text(json.dumps(same, indent=2), encoding="utf-8")
    (OUT / "REQUESTS.json").write_text(json.dumps({
        "schema": "w17-rendered-compositions/requests/1", "STATUS": "RENDERED",
        "written_utc": _now(), "seed": seed, "selection": sel, "requests": requests,
        "reproduction": repro, "declared": declared, "deletion": del_info}, indent=2),
        encoding="utf-8")
    status = {"STATUS": "RENDERED", "written_utc": _now(), "files": figs,
              "declared": declared, "same_decoder": same}
    (OUT / "STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def deletion_panel(objs, saved, od, grid, cond, bitmap, stamps, requests):
    from chlu.core.downdate_store import carry_address_book
    from chlu.eval.rivals.w12_deletion_baselines import load_comparator
    from chlu.experiments import exp_gen_factor_wells as fw
    from chlu.experiments import exp_gen_read as R
    from chlu.experiments.exp_gen_delete_row7 import assert_generation2
    import w12_crossover_lib as L
    from w12_crossover_lib import pinned_basis
    from w12_deletion_rows import (RED_FACTOR, RED_INDEX, SHAPE_FACTOR, SPHERE_INDEX,
                                   factor_indices, match_basis_width)

    seed = S.RENDER_SEED
    L.PART_TEST = str(STAGE / "hub-factor-wells" / "run" / "partitions.npz")
    man = assert_generation2(seed)
    # the staged cache must be byte-identical to the one the deletion rows were run on
    import hashlib
    fp = fw.features_seed_dir(seed) / "features.npy"
    h = hashlib.sha256()
    with open(fp, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    want_sha = fw.feature_cache_manifest(seed)["features_sha256"]
    if h.hexdigest() != want_sha:
        raise SystemExit(f"{fp} sha256 {h.hexdigest()[:16]} != manifest {want_sha[:16]}")
    man["features_sha256_verified"] = want_sha
    _z, _dead, pool = L.load_partition("test")
    cards = [int(c) for c in grid.cardinalities]
    idx = factor_indices(pool, cards)
    m = (idx[:, RED_FACTOR] == RED_INDEX) & (idx[:, SHAPE_FACTOR] == SPHERE_INDEX)
    ev = json.loads((S.EVIDENCE / S.DELETION).read_text(encoding="utf-8"))
    want = int(ev["scopes"]["conjunction_live"]["n_rows_deleted"])
    if int(m.sum()) != want:
        raise SystemExit(f"red-sphere mask has {int(m.sum())} rows, deletion.json says {want}")
    cand = np.flatnonzero(m)
    si, oi = grid.names.index("scale"), grid.names.index("orientation")
    key = np.lexsort((pool[cand], -idx[cand, oi], -idx[cand, si]))
    rid = int(pool[cand[key[0]]])
    vals = grid.values_of(np.array([rid], np.int64))

    F = fw.features_mmap(seed)
    X = np.asarray(fw.gather(F, pool), np.float32)
    st2 = copy.deepcopy(saved["store"])
    U_pre = pinned_basis(st2)
    st2.delete_rows(X[m])
    U_post_raw = pinned_basis(st2)
    U_post, _ = match_basis_width(U_pre, U_post_raw)
    addr2, carry = carry_address_book(saved["address"], U_pre, U_post)
    clu_after = dict(objs[S.CLU], store=st2, basis=U_post_raw, address=addr2,
                     arm="cov_tensor_after_conjunction_live")

    # the downdated CLU must reproduce the banked AFTER counts (deletion.json) on the
    # corners this figure already shows, or the after-panel is not the scored object
    corners = {c["corner_id"]: c for c in fw.pair_subcorners(cond) + fw.nway_corners(cond)}
    banked = {c["corner_id"]: int(c["n_valid"])
              for c in ev["scopes"]["conjunction_live"]["after"]["per_corner"]}
    del_repro = {}
    for cid in S.select(seed)["set1_clu_composes_rivals_fail"]["corners"]:
        cv = fw._request_values_from_rows(grid, corners[cid]["rows"])
        ft, oo = R.arm_features(clu_after, cv, grid)
        ac = fw.accept_rule(cv, oo, np.zeros(cv.shape[0], bool), bitmap, grid)
        im, _st = od.render(ft, arm="__after_repro__")
        ok, _ = od.verdict(cv, ac, im, grid)
        del_repro[cid] = {"n_valid_rendered": int(ok.sum()), "n_valid_banked": banked[cid]}
        if int(ok.sum()) != banked[cid]:
            raise SystemExit(f"DOWNDATE REPRODUCTION FAILED on {cid}: {int(ok.sum())} != "
                             f"deletion.json after {banked[cid]}")

    croot = assemble_comparators(seed)
    before = [objs[S.CLU]] + [objs[f"{f}__as_trained"] for f, _, _ in DELETION_ARMS]
    after = [clu_after] + [
        {"arm": f"{f}__retrain_without__{CONJ}", "read": "rival",
         "rival": load_comparator(seed, f"{f}__retrain_without__{CONJ}", root=croot),
         "rival_partial": None, "store": None, "head_bytes": {"head_bytes": 0}}
        for f, _, _ in DELETION_ARMS]
    imgs, verd, info = [], [], {"request_dataset_row": rid,
                                "factor_names": list(grid.names),
                                "factor_values": [float(x) for x in vals[0]],
                                "n_rows_deleted": int(m.sum()), "feature_cache": man,
                                "carry": {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                                          for k, v in (carry or {}).items()}
                                if isinstance(carry, dict) else str(carry),
                                "verdicts": {},
                                "after_downdate_reproduction": del_repro,
                                "source": str(S.EVIDENCE / S.DELETION)}
    for tag, objs_row in (("before", before), ("after", after)):
        ri, rv = [], []
        for o in objs_row:
            feats, oor = R.arm_features(o, vals, grid)
            acc = fw.accept_rule(vals, oor, np.zeros(1, bool), bitmap, grid)
            im, st = od.render(feats, arm=o["arm"])
            ok, tup = od.verdict(vals, acc, im, grid)
            stamps.extend(st)
            ri.append(im[0])
            rv.append(bool(ok[0]))
            info["verdicts"][f"{tag}:{o['arm']}"] = {"valid": bool(ok[0]),
                                                     "judged_tuple": [int(x) for x in tup[0]]}
        imgs.append(ri)
        verd.append(rv)
    requests.append({"set": "deletion", "corner_id": "red-sphere fit row",
                     "dataset_row": rid, "factor_names": list(grid.names),
                     "factor_values": [float(x) for x in vals[0]], "seed": seed,
                     "verdict": info["verdicts"]})
    dl = ["CLU"] + [l for _, l, _ in DELETION_ARMS]
    dc = ["#2a78d6"] + [c for _, _, c in DELETION_ARMS]
    paths = draw_grid(
        OUT / FIG["deletion"], dl, dc,
        ["before deletion\n(CLU; rival as trained)",
         "after deletion\n(CLU downdate;\nrival retrained without)"],
        imgs, verd, title="Deleted fit row (red sphere): " + _label(grid, vals[0]).replace("\n", ", "))
    return paths, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--placeholder", action="store_true")
    g.add_argument("--render", action="store_true")
    ap.add_argument("--no-nway-scan", action="store_true")
    ap.add_argument("--skip-deletion", action="store_true")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    st = placeholder() if a.placeholder else render(not a.no_nway_scan, a.skip_deletion)
    print(json.dumps(st, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
