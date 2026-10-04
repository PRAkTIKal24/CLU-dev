"""PW-1 item 16.2: n-CLU successes on door transitions, found BY RULE in the persisted per-item scores.

Reads data/cmazet_items.json (base n-CLU, seeds 0-2: the only base seeds with persisted objects)
and data/cmazet_items_rel.json (n-CLU-REL, seeds 0-5). Loads no model; no scoring.

Classes, per model, per seed, over the held-out TEST n-way rows:
  door_open          truth key event > k (a key consumed at a door: the door opens) AND n-CLU correct
  door_then_tint     family == "interaction" (door colour = pocket colour = target tint) AND n-CLU correct
  most_baselines_wrong  n-CLU correct AND >= 4 of the 6 matched baselines wrong
                     (rival: {mlp,lstm,transformer} at 8,000 and 80,000 steps, the run's own objects)
The draw: numpy default_rng(DRAW_SEED), one generator, calls in the fixed order of ALLOCATION.
Within a draw, the qualifying items are grouped into strata (family, colour = pocket colour); the
strata are visited in a random permutation and one item is drawn uniformly inside each stratum, so
consecutive draws differ in family or colour wherever the set allows it. A draw never repeats an item.
An empty class is recorded as empty; nothing is substituted.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DRAW_SEED = 16
K = 4
MATCHED = ("rival: mlp", "rival: lstm", "rival: transformer",
           "rival@10x-steps: mlp", "rival@10x-steps: lstm", "rival@10x-steps: transformer")
REL_COLS = tuple("diagnostic+rel@%s: %s +rel" % (b, f) for b in ("8k", "80k")
                 for f in ("mlp", "lstm", "transformer"))
FILES = {"n-CLU": "cmazet_items.json", "n-CLU-REL": "cmazet_items_rel.json"}
CLASSES = ("door_open", "door_then_tint", "most_baselines_wrong")
#: (figure slot, model, class, how many, per-direction?) -- drawn in this order from ONE generator
ALLOCATION = (
    ("fig_cmazet_families: door row", "door_then_tint", 1, False),
    ("fig_cmazet_families: most-baselines-wrong row", "most_baselines_wrong", 1, False),
    ("fig_cmazet_rel_per_arm: n-CLU-REL success row", "door_then_tint", 1, False),
    ("fig_cmazet_door_then_tint: success rows", "door_then_tint", None, True),
)
#: which model each slot draws from: the figure's own model first; the families figure's door
#: row falls back to n-CLU-REL only because the base n-CLU class is empty (recorded, see search())
SLOT_MODEL = {"fig_cmazet_families: most-baselines-wrong row": ("n-CLU",),
              "fig_cmazet_families: door row": ("n-CLU", "n-CLU-REL"),
              "fig_cmazet_rel_per_arm: n-CLU-REL success row": ("n-CLU-REL",),
              "fig_cmazet_door_then_tint: success rows": ("n-CLU", "n-CLU-REL")}


def classify(srec):
    V, its = srec["verdicts"], srec["items"]
    out = {c: [] for c in CLASSES}
    for j, it in enumerate(its):
        assert it["i"] == j
        if not V["CLU"][j][0]:
            continue
        if it["truth"][2] > K:
            out["door_open"].append(j)
        if it["family"] == "interaction":
            out["door_then_tint"].append(j)
        if sum(1 - V[a][j][0] for a in MATCHED) >= 4:
            out["most_baselines_wrong"].append(j)
    return out


def _stratum(it):
    return (it["family"], int(it["words"][0]))


def _draw(rng, pool, n, taken):
    """pool: [(model, seed, j, item)]. Stratified draw of n items not in ``taken``."""
    pool = [p for p in pool if (p[0], p[1], p[2]) not in taken]
    strata = sorted({_stratum(p[3]) for p in pool})
    got = []
    while len(got) < n and pool:
        order = [strata[int(x)] for x in rng.permutation(len(strata))]
        for s in order:
            cand = [p for p in pool if _stratum(p[3]) == s]
            if not cand or len(got) >= n:
                continue
            p = cand[int(rng.integers(len(cand)))]
            got.append(p)
            pool.remove(p)
        strata = sorted({_stratum(p[3]) for p in pool})
    return got


def search():
    data = {m: json.load(open(os.path.join(HERE, "data", f))) for m, f in FILES.items()}
    counts, qual = {}, {}
    for m, d in data.items():
        counts[m] = {}
        for srec in d["seeds"]:
            s = srec["seed"]
            c = classify(srec)
            its = srec["items"]
            n_door = sum(1 for it in its if it["truth"][2] > K)
            n_int = sum(1 for it in its if it["family"] == "interaction")
            counts[m][str(s)] = {
                "n_nway_rows": len(its),
                "door_open": [len(c["door_open"]), n_door],
                "door_then_tint": [len(c["door_then_tint"]), n_int],
                "door_open_equals_door_then_tint": c["door_open"] == c["door_then_tint"],
                "most_baselines_wrong": len(c["most_baselines_wrong"]),
                "most_baselines_wrong_by_family": {
                    f: sum(1 for j in c["most_baselines_wrong"] if its[j]["family"] == f)
                    for f in sorted({its[j]["family"] for j in c["most_baselines_wrong"]})},
                "n_CLU_correct_and_ge4_of_6_REL_columns_wrong": (
                    sum(1 for j in range(len(its)) if srec["verdicts"]["CLU"][j][0]
                        and sum(1 - srec["verdicts"][a][j][0] for a in REL_COLS) >= 4)
                    if REL_COLS[0] in srec["verdicts"] else None)}
            for cl in CLASSES:
                qual.setdefault((m, cl), []).extend((m, s, j, its[j]) for j in c[cl])
    rng = np.random.default_rng(DRAW_SEED)
    taken, drawn = set(), []
    for slot, cl, n, per_dir in ALLOCATION:
        model = next((m for m in SLOT_MODEL[slot] if qual[(m, cl)]), None)
        empty = [m for m in SLOT_MODEL[slot] if not qual[(m, cl)]]
        rec = dict(slot=slot, cls=cl, empty_for=empty, model=model, items=[])
        if model is not None:
            pool = qual[(model, cl)]
            if per_dir:
                for dr in "NESW":
                    sub = [p for p in pool if p[3]["action"] == dr]
                    rec["items"] += _draw(rng, sub, 1, taken)
            else:
                rec["items"] = _draw(rng, pool, n, taken)
            for p in rec["items"]:
                taken.add((p[0], p[1], p[2]))
            rec["n_qualifying"] = len(pool)
        rec["items"] = [dict(model=p[0], seed=p[1], i=p[2], family=p[3]["family"],
                             colour=int(p[3]["words"][0]), action=p[3]["action"],
                             truth=p[3]["truth"]) for p in rec["items"]]
        drawn.append(rec)
    return dict(rule=__doc__.strip(), draw_seed=DRAW_SEED, k=K, matched_baselines=list(MATCHED),
                base_nclu_seeds_searched=[0, 1, 2],
                base_nclu_seeds_not_searchable={"seeds": [3, 4, 5], "why": (
                    "no persisted base n-CLU objects for seeds 3-5 (~/clu/stage/w19-persist holds "
                    "BOTH_seed0..2 only), so no per-item scores exist; nothing was re-scored")},
                counts=counts, drawn=drawn,
                items_sha256={m: __import__("hashlib").sha256(
                    open(os.path.join(HERE, "data", f), "rb").read()).hexdigest()
                    for m, f in FILES.items()})


def picks(res, slot):
    return next(r for r in res["drawn"] if r["slot"] == slot)


if __name__ == "__main__":
    print(json.dumps(search(), indent=1, sort_keys=True))
