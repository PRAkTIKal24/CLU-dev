"""CLAIM 1 -- independent re-derivation of the frozen exact-match column.

Reads only the three frozen Mac cell manifests. No hub scripts imported.
"""
import json, os, math, hashlib, statistics, sys

BASE = r"C:\Users\PratikJawahar\chlu\chlu-bundle-v2\.claude\outputs\gen-read-cell-arm\runs\cell"
SEEDS = (0, 1, 2)
OUTDIR = os.path.join(os.environ["OUT"].replace("/c/", "C:/").replace("/", os.sep), "replication") \
    if os.environ.get("OUT", "").startswith("/c/") else os.path.join(os.environ["OUT"], "replication")


def manifest_path(s):
    return os.path.join(BASE, f"seed_{s}", f"seed_{s}", "cell_manifest.json")


def sd(xs):
    # sample sd (n-1), matching the usual "mean +/- sd over 3 seeds" convention
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def sd_pop(xs):
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def main():
    mans, digests = {}, {}
    for s in SEEDS:
        p = manifest_path(s)
        raw = open(p, "rb").read()
        digests[s] = hashlib.sha256(raw).hexdigest()[:16]
        mans[s] = json.loads(raw.decode("utf-8"))

    arms = sorted(set().union(*[set(m["arms"]) for m in mans.values()]))
    rows = []
    for arm in arms:
        em, ft, nB = [], [], []
        for s in SEEDS:
            a = mans[s]["arms"].get(arm)
            if a is None:
                em.append(None); ft.append(None); nB.append(None); continue
            cb = a["external"]["composition_B"]
            em.append(cb["exact_match"]); nB.append(cb["n"])
            ft.append(a["distinct"]["B"]["factor_tuples"])
        cnt = [None if e is None else e * 72 for e in em]
        good = [c for c in cnt if c is not None]
        rows.append(dict(
            arm=arm, exact_match=em, count72=cnt, n_B=nB, factor_tuples=ft,
            mean=statistics.fmean(good) if good else None,
            sd_sample=sd(good) if good else None,
            sd_pop=sd_pop(good) if good else None,
        ))

    ce_em = [mans[s]["ceiling"]["external"]["composition_B"]["exact_match"] for s in SEEDS]
    ce_cnt = [e * 72 for e in ce_em]
    ceiling = dict(arm="__ceiling__", exact_match=ce_em, count72=ce_cnt,
                   n_B=[mans[s]["ceiling"]["external"]["composition_B"]["n"] for s in SEEDS],
                   mean=statistics.fmean(ce_cnt), sd_sample=sd(ce_cnt), sd_pop=sd_pop(ce_cnt))

    out = dict(manifest_sha256_16=digests,
               manifest_paths={s: manifest_path(s) for s in SEEDS},
               ceiling=ceiling, arms=rows)

    w = max(len(r["arm"]) for r in rows) + 2
    print(f"{'arm':<{w}} {'s0':>7} {'s1':>7} {'s2':>7} | {'mean':>7} {'sd(n-1)':>8} {'sd(n)':>7} | factor_tuples")
    for r in [ceiling] + rows:
        c = r["count72"]
        f = r.get("factor_tuples")
        print(f"{r['arm']:<{w}} " + " ".join(f"{x:7.3f}" if x is not None else "   None" for x in c)
              + f" | {r['mean']:7.3f} {r['sd_sample']:8.3f} {r['sd_pop']:7.3f} | {f}")
    return out


if __name__ == "__main__":
    res = main()
    dest = sys.argv[1] if len(sys.argv) > 1 else None
    if dest:
        json.dump(res, open(dest, "w"), indent=1)
        print("wrote", dest)
