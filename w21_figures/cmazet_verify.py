"""Part C gate 1: assert the hashes of every staged w19 persisted object before anything is loaded.

For every <stage>/BOTH_seed{0,1,2}/{test,val}/:
  * each object's .sha256 carries exactly two lines: `param_sha256 <h>` and `file_sha256 <h>  <name>`;
  * file_sha256 == SHA-256 of the file on disk;
  * the object is in MANIFEST.json with the same file_sha256, the same param_sha256 and
    bytes == the file's size;
  * every MANIFEST entry has its file and every non-.sha256, non-MANIFEST file is in MANIFEST.
A single mismatch exits 1 (STOP: nothing is drawn). Writes a JSON report.
usage: python cmazet_verify.py <stage root> <report.json> [--seeds 0,1,2] [--legs test,val]
(w22: seeds/legs are arguments so the CLU-REL-A10 stage, test only, seeds 0-5, runs the same gate.)
"""
import hashlib
import json
import os
import sys


def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("report")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--legs", default="test,val")
    a = ap.parse_args()
    root, rep_path = a.root, a.report
    seeds = [int(x) for x in a.seeds.split(",")]
    legs = a.legs.split(",")
    report, bad, n_ok = {}, [], 0
    for seed in seeds:
        for leg in legs:
            d = os.path.join(root, f"BOTH_seed{seed}", leg)
            if not os.path.isdir(d):
                bad.append(f"missing dir {d}")
                continue
            man = json.load(open(os.path.join(d, "MANIFEST.json")))
            files = sorted(f for f in os.listdir(d)
                           if not f.endswith(".sha256") and f != "MANIFEST.json")
            for f in files:
                if f not in man:
                    bad.append(f"{d}/{f}: not in MANIFEST")
            for name, m in sorted(man.items()):
                p = os.path.join(d, name)
                if not os.path.exists(p):
                    bad.append(f"{p}: in MANIFEST, missing on disk")
                    continue
                lines = open(p + ".sha256").read().strip().split("\n")
                if len(lines) != 2 or not lines[0].startswith("param_sha256 ") \
                        or not lines[1].startswith("file_sha256 "):
                    bad.append(f"{p}.sha256: not the two-line format: {lines!r}")
                    continue
                psha = lines[0].split()[1]
                parts = lines[1].split()
                ffile, fname = parts[1], parts[2]
                got = fsha(p)
                size = os.path.getsize(p)
                checks = {
                    "file_sha256 == disk": ffile == got,
                    "sha256 line names the file": fname == name,
                    "MANIFEST file_sha256 == disk": m["file_sha256"] == got,
                    "MANIFEST param_sha256 == .sha256": m["param_sha256"] == psha,
                    "MANIFEST bytes == size": m["bytes"] == size,
                }
                for k, v in checks.items():
                    if not v:
                        bad.append(f"{p}: FAIL {k}")
                n_ok += all(checks.values())
                report[f"BOTH_seed{seed}/{leg}/{name}"] = dict(
                    file_sha256=got, param_sha256=psha, bytes=size, ok=all(checks.values()),
                    rebuild=m.get("rebuild"))
    out = dict(stage_root=root, n_objects=len(report), n_ok=n_ok, mismatches=bad,
               verdict="ALL HASHES MATCH" if not bad else "MISMATCH - STOP", objects=report)
    json.dump(out, open(rep_path, "w"), indent=1, sort_keys=True)
    print(out["verdict"], f"{n_ok}/{len(report)} objects", *bad[:20], sep="\n")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
