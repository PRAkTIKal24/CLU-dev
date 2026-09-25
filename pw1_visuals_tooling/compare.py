#!/usr/bin/env python3
"""Compare a build's outputs (its SHA256 manifest) with the installed files of paper-visuals/ and
w17-paper/figures/, and diff PROVENANCE.json structurally. Prints a markdown table.
usage: compare.py <build dest dir> [<installed dir> ...]"""
import hashlib
import json
import os
import sys

PLACED = ["fig_task_cmazet", "fig_ladder", "fig_crossover", "fig_deletion", "fig_encoder_adapt",
          "fig_maze_conditions", "fig_cmazet_families", "fig_cmazet_per_arm",
          "fig_cmazet_rel_per_arm", "fig_cmazet_door_then_tint"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.isfile(p) else None


def flat(o, pre=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flat(v, pre + "/" + str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from flat(v, pre + "[%d]" % i)
    else:
        yield pre, o


def main(build, *inst):
    man = [l.split() for l in open(os.path.join(build, "SHA256-w22-build1.txt"))]
    man2 = [l.split() for l in open(os.path.join(build, "SHA256-w22-build2.txt"))]
    print("build a == build b (manifests):", man == man2, "| files:", len(man))
    names = [n for _, n in man]
    hdr = "| file | build a | " + " | ".join(os.path.basename(os.path.dirname(i + "/")) + " (installed)"
                                          for i in inst) + " | equal |"
    print(hdr)
    print("|" + "---|" * (3 + len(inst)))
    n_eq = 0
    for h, n in man:
        hs = [sha(os.path.join(i, n)) for i in inst]
        eq = all(x == h for x in hs)
        n_eq += eq
        print("| %s | %s | %s | %s |" % (n, h[:16], " | ".join((x or "MISSING")[:16] for x in hs),
                                         "YES" if eq else "**NO**"))
    print("\nequal: %d / %d" % (n_eq, len(man)))
    stems = sorted({os.path.splitext(n)[0] for n in names if n.startswith("fig_")})
    print("stems built:", len(stems), stems)
    print("placed build stems (10 of the 13 placed; the other 3 are mirrored):",
          all(s in stems for s in PLACED))
    for i in inst:
        a = json.load(open(os.path.join(build, "PROVENANCE.json")))
        b = json.load(open(os.path.join(i, "PROVENANCE.json")))
        fa, fb = dict(flat(a)), dict(flat(b))
        only_b = sorted(k for k in fb if k not in fa)
        only_a = sorted(k for k in fa if k not in fb)
        chg = sorted(k for k in fa if k in fb and fa[k] != fb[k])
        roots = sorted({"/".join(k.split("/")[:3]) for k in only_b})
        print("\nPROVENANCE.json vs %s: leaves only installed %d (under %s), only built %d %s, "
              "changed %d" % (i, len(only_b), roots, len(only_a), only_a[:10], len(chg)))
        for k in chg:
            print("  changed %s: installed %r -> built %r" % (k, str(fb[k])[:80], str(fa[k])[:80]))
        other = [k for k in only_b if not k.startswith("/figures/fig_headline_steps")]
        for k in other:
            print("  only installed (not headline): %s = %r" % (k, str(fb[k])[:80]))


if __name__ == "__main__":
    main(*sys.argv[1:])
