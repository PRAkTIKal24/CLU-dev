"""Independent check of REQUEST-LAYER-READY.json.  A published file is not a finished file.

Usage:  python -m w14_request_layer.verify <out_dir>

This does NOT import ``w14_request_layer.readiness`` and does not read the measure JSONs.
It goes back to the bank pickles and to the engine, recomputes the load-bearing numbers a
second way, and compares them with what the readiness file claims.  Exit code 0 only if
every check passes; every check prints its own line.

Checks:
  1  the key-count histogram in the file equals one recomputed from the bank pickles by
     replaying every stored route through ``w8_envscout.screen.key_order_of``;
  2  every stored route is accepted by the EXACT judge at budget = opt (no slack), and its
     length equals the stored optimum -- so the oracle's 1.0000 is not a reporting artefact;
  3  no stored route picks up fewer than 2 keys in an AFTER bank;
  4  the judge's file on disk still has the sha256 the readiness file recorded, and the
     judge object is ``w8_envscout.maze_colour.judge`` itself;
  5  ``tolerance_used`` is null everywhere it appears in the file;
  6  for every kept pair the doors are LOAD-BEARING -- the target is unreachable with no
     key at all -- and the agent word and the target word never share a door-free
     component of the site's OWN grid (which on a walled site carries the seal).
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys

from w8_envscout.maze_colour import _reachable_without_keys, judge
from w8_envscout.screen import key_order_of
from w9_maze.task import door_free_components

OK, BAD = [], []


def check(name, cond, detail=""):
    (OK if cond else BAD).append(name)
    print("  %-4s %-52s %s" % ("PASS" if cond else "FAIL", name, detail))


def _walk(o, key):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key:
                yield v
            yield from _walk(v, key)
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v, key)


def main(out_dir):
    ready = json.load(open(os.path.join(out_dir, "REQUEST-LAYER-READY.json")))
    banks = ready["provenance"]["banks"]

    # 4 -- the judge has not moved
    jd = ready["provenance"]["judge"]
    path = os.path.join(os.getcwd(), jd["source_file"])
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    check("judge file sha256 unchanged", sha == jd["sha256"], sha[:16])
    check("judge is maze_colour.judge itself",
          judge.__module__ == jd["module"] and judge.__qualname__ == jd["qualname"])

    # 5 -- no tolerance anywhere
    tols = list(_walk(ready, "tolerance_used"))
    check("tolerance_used is null everywhere", bool(tols) and all(t is None for t in tols),
          "%d occurrences" % len(tols))

    for name, meta in sorted(banks.items()):
        blob = pickle.load(open(meta["path"], "rb"))
        sites = blob["sites"]
        hist, judged, badlen, minkeys = {}, 0, 0, 99
        for st in sites:
            for (i, j) in st.pairs:
                route = st.route[(i, j)]
                inst = st.instance(i, j, st.opt[(i, j)])
                k = len(key_order_of(inst, route))
                hist[k] = hist.get(k, 0) + 1
                minkeys = min(minkeys, k)
                ok, _reason, steps = judge(inst, route)
                judged += bool(ok)
                badlen += (len(route) != st.opt[(i, j)] or steps != st.opt[(i, j)])
        n = sum(hist.values())
        claimed = ready["KEY_COUNT_HISTOGRAM"][
            "AFTER__w14_request_layer" if meta["path"].find("after") >= 0
            else "BEFORE__w9_request_layer"]
        check("%s: n_routes matches" % name, n == meta["n_routes"],
              "%d" % n)
        check("%s: every stored route passes the EXACT judge at budget=opt" % name,
              judged == n, "%d/%d" % (judged, n))
        check("%s: stored route length == stored optimum" % name, badlen == 0,
              "%d mismatches" % badlen)
        if "after" in meta["path"]:
            check("%s: no route uses fewer than 2 keys" % name, minkeys >= 2,
                  "min keys = %d" % minkeys)
        # 6 -- the doors are load-bearing for every kept pair, and the pair's two words
        #      never share a door-free component of the site's OWN (possibly sealed) grid.
        #      NB the components must be recomputed on ``st.grid``: on a site with a
        #      walled target that grid carries the seal, so it is not ``generate()``'s.
        bad_reach = bad_comp = 0
        for st in sites:
            comps = door_free_components(st.grid, st.door)
            comp_of = {c: idx for idx, comp in enumerate(comps) for c in comp}
            for (i, j) in st.pairs:
                if _reachable_without_keys(st.instance(i, j, 10 ** 9)):
                    bad_reach += 1
                if comp_of.get(st.agents[i], -1) == comp_of.get(st.targets[j], -2):
                    bad_comp += 1
        check("%s: the doors are load-bearing for every kept pair" % name,
              bad_reach == 0, "%d pairs reachable with no key" % bad_reach)
        check("%s: agent and target never share a door-free component" % name,
              bad_comp == 0, "%d pairs violate" % bad_comp)
        # 1 -- histogram agreement (summed over banks of the same kind is checked below)
        meta["_hist"] = hist

    for kind, key in (("after", "AFTER__w14_request_layer"),
                      ("before", "BEFORE__w9_request_layer")):
        merged = {}
        for meta in banks.values():
            if kind in meta["path"] and "_hist" in meta:
                for k, v in meta["_hist"].items():
                    merged[str(k)] = merged.get(str(k), 0) + v
        claimed = ready["KEY_COUNT_HISTOGRAM"][key]["histogram"]
        check("%s histogram recomputed == published" % kind.upper(),
              merged == {k: v for k, v in claimed.items()},
              "%s vs %s" % (merged, claimed))

    print()
    print("%d passed, %d FAILED" % (len(OK), len(BAD)))
    return 0 if not BAD else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
