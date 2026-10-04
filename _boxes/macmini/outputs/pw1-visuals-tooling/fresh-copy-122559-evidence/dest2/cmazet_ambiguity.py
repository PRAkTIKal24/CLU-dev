"""Count same-picture / opposite-mark arm pairs in the CMAZE-T per-item re-read.

A PICTURE is everything a panel shows for one arm on one item. Two arms on the same item whose
pictures are identical but whose judge marks differ are a figure defect: the reader cannot tell
from the image why one is right. Counted under successive picture definitions:
  state         -- the drawn generated s_{t+1} only (gen.pos, gen.ks)           [pass 1/2 panels]
  state+legal   -- plus the emitted legality                                      [pass 3]
  state+legal+pocket -- plus the emitted pocket-next
  full          -- the whole emitted fact vector (sanity: must be 0 by construction)
  pixels        -- the RENDERED panel image (draw_state RGB bytes): what pass 1/2 actually showed
  pixels+strip  -- the rendered image plus the fact strip's displayed tuple (facts_shown): pass 3
Reads data/cmazet_items.json only; loads no model; no scoring.
"""
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cmazet_panels as P  # noqa: E402

DEFS = ("state", "state+legal", "state+legal+pocket", "full", "pixels", "pixels+strip")


def key(defn, gen, em, px=None):
    if defn == "pixels":
        return px
    if defn == "pixels+strip":
        return (px,) + P.facts_shown(em)
    s = (tuple(gen.pos), tuple(gen.ks))
    if defn == "state":
        return s
    if defn == "state+legal":
        return s + (int(em[0]),)
    if defn == "state+legal+pocket":
        return s + (int(em[0]), int(em[3]))
    return tuple(int(x) for x in em)


def count(items_path, arm_filter=None, win=5):
    d = json.load(open(items_path))
    out = {"per_seed": {}, "by_reason_pair": {}}
    tot = {k: 0 for k in DEFS}
    htot = {"state": 0, "pixels": 0, "pixels+strip": 0}
    opp_total = 0
    for srec in d["seeds"]:
        mz = P.Maze(srec)
        arms = [a for a in srec["verdicts"] if arm_filter is None or a in arm_filter]
        c = {k: 0 for k in DEFS}
        opp = 0
        hc = {"state": 0, "pixels": 0, "pixels+strip": 0}
        for it in srec["items"]:
            i = it["i"]
            rec = {}
            for a in arms:
                ok, reason, em = srec["verdicts"][a][i]
                inst, st, gen = mz.states(it, em)
                px = P.draw_state(inst, gen.pos, gen.ks, st.pos, win).tobytes()
                rec[a] = (bool(ok), reason, em, gen, px)
            for a, b in itertools.combinations(arms, 2):
                if rec[a][0] == rec[b][0]:
                    continue
                opp += 1
                ea, eb = rec[a][2], rec[b][2]
                if ea[1:] == eb[1:] and ea[0] != eb[0]:
                    for defn in ("state", "pixels", "pixels+strip"):
                        if (key(defn, rec[a][3], ea, rec[a][4])
                                == key(defn, rec[b][3], eb, rec[b][4])):
                            hc[defn] += 1
                for defn in DEFS:
                    if (key(defn, rec[a][3], rec[a][2], rec[a][4])
                            == key(defn, rec[b][3], rec[b][2], rec[b][4])):
                        c[defn] += 1
                        if defn == "pixels":
                            bad = rec[a][1] if not rec[a][0] else rec[b][1]
                            out["by_reason_pair"][bad] = out["by_reason_pair"].get(bad, 0) + 1
        for k in hc:
            htot[k] += hc[k]
        out["per_seed"][srec["seed"]] = dict(opposite_mark_pairs=opp, same_picture=c,
                                             head_rule_legal_bit_only=hc,
                                             n_items=len(srec["items"]), n_arms=len(arms))
        opp_total += opp
        for k in DEFS:
            tot[k] += c[k]
    out["total"] = dict(opposite_mark_pairs=opp_total, same_picture=tot,
                        head_rule_legal_bit_only=htot)
    return out


if __name__ == "__main__":
    import hashlib
    spec = json.load(open(os.path.join(HERE, "data", "cmazet_panels.json")))
    drawn = {a["key"] for a in spec["arms"]}
    rel = next(p for p in spec["panel_sets"] if p["stem"] == "fig_cmazet_rel_per_arm")
    res = {"head_count_rule": "pairs with identical picture whose emissions differ ONLY in the "
                              "legal bit"}
    for tag, fn, dr in (("A1_w19", "cmazet_items.json", drawn),
                        ("CLU_REL_A10", "cmazet_items_rel.json",
                         drawn | {a["key"] for a in rel.get("extra_arms", [])})):
        items = os.path.join(HERE, "data", fn)
        res[tag] = {"items_file": "data/" + fn,
                    "items_sha256": hashlib.sha256(open(items, "rb").read()).hexdigest(),
                    "all_arms_in_items_file": count(items),
                    "drawn_arms_only": count(items, dr)}
    print(json.dumps(res, indent=1))
