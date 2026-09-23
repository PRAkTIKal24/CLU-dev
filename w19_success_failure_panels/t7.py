"""T7 (Shapes3D words -> image, K* = 32) success/failure panels, from the per-request scan.

SELECTION RULES (applied to scan/verdicts.json; nothing is hand-picked)
* per-arm   for the CLU and each of the four rivals: one held-out seed-0 request the arm
            renders VALIDLY and one it renders INVALIDLY, each drawn uniformly at random
            (numpy default_rng(SELECT_SEED), arms in column order, success then failure)
            from ALL qualifying seed-0 held-out requests.  All six columns are rendered on
            every row, so each row also shows every other arm on the same request.
* rival-succeeds-CLU-fails
            every held-out request at seeds 0, 1, 2 where a rival is valid and the CLU is
            not is COUNTED (per seed, per rival).  One per rival is drawn (same rng rule)
            at the lowest seed where that rival has any; a rival with none at any seed is
            DECLARED.  Each seed renders through its own hash-asserted decoder, one figure
            per seed, so a figure never mixes seeds.
Set 1 and set 3 are w17's and are not rebuilt here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import SPOKE_OUT, w17

SELECT_SEED = 19
RIVALS = ("mlp__as_trained", "lstm__as_trained", "transformer__as_trained",
          "cont_mlp__as_trained")
FIG_PER_ARM = "rc_w19_t7_per_arm_success_failure"
FIG_RSCF = "rc_w19_t7_rival_succeeds_clu_fails"


def load_scan():
    p = SPOKE_OUT / "scan" / "verdicts.json"
    return json.loads(p.read_text(encoding="utf-8")), p


def counts(scan) -> dict:
    arms = scan["arms"]
    ci = arms.index("cov_tensor")
    out = {}
    for sd, d in scan["seeds"].items():
        n = len(d["requests"])
        per = {}
        for a in RIVALS + ("table",):
            j = arms.index(a)
            per[a] = sum(1 for r in d["requests"] if r[4][j] == "1" and r[4][ci] == "0")
        any_r = sum(1 for r in d["requests"]
                    if r[4][ci] == "0" and any(r[4][arms.index(a)] == "1" for a in RIVALS))
        valid = {a: sum(1 for r in d["requests"] if r[4][arms.index(a)] == "1") for a in arms}
        out[sd] = {"n_heldout_requests": n, "n_valid_per_arm": valid,
                   "rival_valid_and_CLU_invalid": per,
                   "any_rival_valid_and_CLU_invalid": any_r,
                   "note": "'table' is the lookup floor, counted for completeness; it is "
                           "a floor, never a rival"}
    return out


def select(scan) -> dict:
    arms = scan["arms"]
    rng = np.random.default_rng(SELECT_SEED)
    reqs0 = scan["seeds"]["0"]["requests"]
    per_arm = []
    for a in ("cov_tensor",) + RIVALS:
        j = arms.index(a)
        for want, tag in (("1", "success"), ("0", "failure")):
            pool = [r for r in reqs0 if r[4][j] == want]
            if not pool:
                per_arm.append({"arm": a, "kind": tag, "DECLARED": "no qualifying seed-0 request"})
                continue
            r = pool[int(rng.integers(len(pool)))]
            per_arm.append({"arm": a, "kind": tag, "seed": 0, "corner_id": r[0], "family": r[1],
                            "dataset_row": r[2], "index_in_corner": r[3], "scan_bits": r[4],
                            "pool_size": len(pool)})
    ci = arms.index("cov_tensor")
    rscf = []
    for a in RIVALS:
        j = arms.index(a)
        for sd in sorted(scan["seeds"], key=int):
            pool = [r for r in scan["seeds"][sd]["requests"] if r[4][j] == "1" and r[4][ci] == "0"]
            if pool:
                r = pool[int(rng.integers(len(pool)))]
                rscf.append({"arm": a, "kind": "rival succeeds, CLU fails", "seed": int(sd),
                             "corner_id": r[0], "family": r[1], "dataset_row": r[2],
                             "index_in_corner": r[3], "scan_bits": r[4], "pool_size": len(pool)})
                break
        else:
            rscf.append({"arm": a, "kind": "rival succeeds, CLU fails",
                         "DECLARED": "no held-out request at seed 0, 1 or 2"})
    return {"select_seed": SELECT_SEED, "per_arm": per_arm, "rival_succeeds_clu_fails": rscf}


class SeedCtx:
    """One seed's judge, ONE hash-asserted decoder, arm objects, grid and corners."""

    def __init__(self, seed):
        RR = w17()
        from w17_rendered_compositions import selection as S
        from chlu.experiments import exp_gen_factor_wells as fw
        fw.PINS["condition"] = "test"
        fw.PINS["seed"] = seed
        cond = fw.load_condition()
        self.seed, self.RR, self.S, self.fw = seed, RR, S, fw
        self.grid = cond["grid"]
        self.bitmap = fw.value_presence_bitmap(self.grid, cond["train_rows"])
        self.corners = {c["corner_id"]: c for c in fw.pair_subcorners(cond) + fw.nway_corners(cond)}
        self.judge = fw.Judge(seed, log=print)
        self.winner = S.EVIDENCE / S.WINNER_HASHES
        self.od = RR.OneDecoder.from_judge(self.judge, seed, self.winner)
        self.objs, _ = RR._load_arms(seed, self.grid, None)
        self.order = [a for a, _, _ in S.ARMS]
        self.stamps = []

    def one(self, sel):
        c = self.corners[sel["corner_id"]]
        i = int(sel["index_in_corner"])
        if int(c["rows"][i]) != int(sel["dataset_row"]):
            raise SystemExit(f"row mismatch {sel}")
        sub = dict(c, rows=np.asarray(c["rows"])[i:i + 1])
        vals, res = self.RR._read_corner(self.objs, self.od, sub, self.grid, self.bitmap)
        bits = "".join("1" if bool(res[a]["valid"][0]) else "0" for a in self.order)
        if bits != sel["scan_bits"]:
            raise SystemExit(f"single-request re-render {bits} != scan {sel['scan_bits']} for {sel}")
        for a in self.order:
            self.stamps.append(res[a]["stamps"][0])
        rec = {"factor_names": list(self.grid.names),
               "factor_values": [float(x) for x in vals[0]],
               "verdict": {a: bool(res[a]["valid"][0]) for a in self.order},
               "accepted": {a: bool(res[a]["accept"][0]) for a in self.order},
               "judged_tuple": {a: [int(x) for x in res[a]["tuples"][0]] for a in self.order},
               "rerender_equals_scan": True}
        return vals[0], res, rec

    def positive_control(self, sels):
        from chlu.experiments import exp_gen_read as R
        rows = []
        for s in sels:
            c = self.corners[s["corner_id"]]
            rows.append(int(c["rows"][int(s["index_in_corner"])]))
        v = self.fw._request_values_from_rows(self.grid, np.asarray(rows))
        f, _ = R.arm_features(self.objs[self.S.CLU], v, self.grid)
        return self.od.positive_control(self.judge, f)

    def assert_same(self):
        from w17_rendered_compositions.same_decoder import assert_same_decoder
        return assert_same_decoder(self.stamps, self.winner, self.seed)


def _row_label(ctx, sel, vrow):
    fam = {"pair": "held-out 2-way corner", "nway": "held-out 6-factor box"}[sel["family"]]
    return fam + "\n" + ctx.RR._label(ctx.grid, vrow)


def build(out_dir: Path) -> dict:
    from .figure import draw_grid
    scan, scan_path = load_scan()
    sel = select(scan)
    from w17_rendered_compositions import selection as S
    labels = [l for _, l, _ in S.ARMS]
    colours = [c for _, _, c in S.ARMS]
    order = [a for a, _, _ in S.ARMS]
    colour_of = {a: c for a, _, c in S.ARMS}
    name_of = {a: l.replace("\n", " ") for a, l, _ in S.ARMS}
    ctxs, figs, same = {}, {}, {}

    def ctx(sd):
        if sd not in ctxs:
            ctxs[sd] = SeedCtx(sd)
        return ctxs[sd]

    # per-arm figure (seed 0)
    rows = [s for s in sel["per_arm"] if "DECLARED" not in s]
    imgs, verd, rl, hl, groups = [], [], [], {}, []
    for r_i, s in enumerate(rows):
        c0 = ctx(0)
        vrow, res, rec = c0.one(s)
        s.update(rec)
        imgs.append([res[a]["imgs"][0] for a in order])
        verd.append([bool(res[a]["valid"][0]) for a in order])
        rl.append(_row_label(c0, s, vrow))
        hl[(r_i, order.index(s["arm"]))] = colour_of[s["arm"]]
    for a in ("cov_tensor",) + RIVALS:
        idx = [i for i, s in enumerate(rows) if s["arm"] == a]
        if idx:
            groups.append((name_of[a], colour_of[a], idx[0], idx[-1]))
    figs["per_arm"] = draw_grid(out_dir / FIG_PER_ARM, labels, colours, rl, imgs, verd,
                                row_groups=groups, highlight=hl)
    # rival succeeds, CLU fails -- one figure per seed involved
    rs = [s for s in sel["rival_succeeds_clu_fails"] if "DECLARED" not in s]
    for sd in sorted({s["seed"] for s in rs}):
        cs = ctx(sd)
        sub = [s for s in rs if s["seed"] == sd]
        imgs, verd, rl, hl, groups = [], [], [], {}, []
        for r_i, s in enumerate(sub):
            vrow, res, rec = cs.one(s)
            s.update(rec)
            imgs.append([res[a]["imgs"][0] for a in order])
            verd.append([bool(res[a]["valid"][0]) for a in order])
            rl.append(_row_label(cs, s, vrow))
            hl[(r_i, order.index(s["arm"]))] = colour_of[s["arm"]]
            hl[(r_i, order.index("cov_tensor"))] = colour_of["cov_tensor"]
            groups.append((name_of[s["arm"]], colour_of[s["arm"]], r_i, r_i))
        stem = FIG_RSCF + ("" if sd == 0 else f"_seed{sd}")
        figs[f"rscf_seed{sd}"] = draw_grid(out_dir / stem, labels, colours, rl, imgs, verd,
                                           row_groups=groups, highlight=hl)
    for sd, cx in sorted(ctxs.items()):
        used = [s for s in sel["per_arm"] + sel["rival_succeeds_clu_fails"]
                if s.get("seed") == sd and "DECLARED" not in s]
        pc = cx.positive_control(used)
        same[str(sd)] = dict(cx.assert_same(), positive_control=pc,
                             classifier_param_hash=cx.od.classifier_hash)
    return {"figures": figs, "selection": sel, "counts": counts(scan),
            "scan": str(scan_path), "same_decoder": same}
