"""Mechanical verification of the w14 deltas. Nothing here is asserted in prose."""
import io, os, re, difflib, hashlib

BASE = r"C:\chlu\evidence\outputs\w13-registry-fold"
OUT  = r"C:\chlu\outputs\w14-refusal-displays-closeout"

def rd(p):
    with io.open(p, "r", encoding="utf-8", newline="\n") as f:
        return f.read()

def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()

print("=" * 78)
print("1. ADDITIVITY — difflib.SequenceMatcher over base vs new, per line")
print("=" * 78)
for name in ("negative_results.md", "claims_matrix.md"):
    a = rd(os.path.join(BASE, name)).split("\n")
    b = rd(os.path.join(OUT,  name)).split("\n")
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    tot = {"equal": 0, "insert": 0, "replace": 0, "delete": 0}
    blocks = {"equal": 0, "insert": 0, "replace": 0, "delete": 0}
    repl = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        blocks[tag] += 1
        tot[tag] += max(i2 - i1, j2 - j1)
        if tag in ("replace", "delete"):
            repl.append((tag, i1, i2, j1, j2))
    print("\n%s : %d -> %d lines" % (name, len(a), len(b)))
    print("  lines  equal=%d insert=%d replace=%d delete=%d" %
          (tot["equal"], tot["insert"], tot["replace"], tot["delete"]))
    print("  blocks equal=%d insert=%d replace=%d delete=%d" %
          (blocks["equal"], blocks["insert"], blocks["replace"], blocks["delete"]))
    for tag, i1, i2, j1, j2 in repl:
        print("    !! %s base[%d:%d] -> new[%d:%d]" % (tag, i1, i2, j1, j2))
        for k in range(i1, i2):
            print("       - " + a[k][:110])

print()
print("=" * 78)
print("2. claims_matrix.md — every line but line 1 byte-identical; prior title preserved")
print("=" * 78)
a = rd(os.path.join(BASE, "claims_matrix.md")).split("\n")
b = rd(os.path.join(OUT,  "claims_matrix.md")).split("\n")
print("  lines 2..%d identical :" % len(a), a[1:] == b[1:len(a)])
print("  new title endswith prior title :", b[0].endswith(a[0]))
print("  title chars %d -> %d" % (len(a[0]), len(b[0])))
print("  appended lines :", len(b) - len(a))

print()
print("=" * 78)
print("3. NUMBER INTEGRITY — no N renumbered, N337 new and unique, one ### per N")
print("=" * 78)
an = rd(os.path.join(BASE, "negative_results.md"))
bn = rd(os.path.join(OUT,  "negative_results.md"))
heads_a = re.findall(r"^### (N[\w\-]+) ", an, re.M)
heads_b = re.findall(r"^### (N[\w\-]+) ", bn, re.M)
print("  base entry headings : %d" % len(heads_a))
print("  new  entry headings : %d" % len(heads_b))
print("  added               :", [h for h in heads_b if h not in heads_a])
print("  removed             :", [h for h in heads_a if h not in heads_b])
dups_b = sorted({h for h in heads_b if heads_b.count(h) > 1})
print("  duplicate headings in new :", dups_b if dups_b else "NONE")
print("  'N337' occurrences base=%d new=%d" % (an.count("N337"), bn.count("N337")))
print("  fold symbol U+2299 base=%d new=%d" % (an.count("\u2299"), bn.count("\u2299")))

am = rd(os.path.join(BASE, "claims_matrix.md"))
bm = rd(os.path.join(OUT,  "claims_matrix.md"))
cm_a = re.findall(r"^### \*\*(CM-[\w\-]+)\*\*", am, re.M)
cm_b = re.findall(r"^### \*\*(CM-[\w\-]+)\*\*", bm, re.M)
print("  base CM rows : %d ; new CM rows : %d" % (len(cm_a), len(cm_b)))
print("  added CM     :", [c for c in cm_b if c not in cm_a])
print("  removed CM   :", [c for c in cm_a if c not in cm_b])
dups_cm = sorted({c for c in cm_b if cm_b.count(c) > 1})
print("  duplicate CM rows in new :", dups_cm if dups_cm else "NONE")

print()
print("=" * 78)
print("4. THE RETIRED ROWS ARE STILL THERE — never deleted")
print("=" * 78)
for probe in ("### N323 ", "### N-I8-19 ", "### **CM-R12-3**", "### N-I8-6 ", "### N332 ", "### N334 "):
    tgt = bn if probe.startswith("### N") else bm
    print("  %-22s present=%s  count=%d" % (probe.strip(), probe in tgt, tgt.count(probe)))

print()
print("=" * 78)
print("5. EVIDENCE UNTOUCHED — md5 re-checked at close")
print("=" * 78)
for p in (r"C:\chlu\evidence\negative_results.md",
          r"C:\chlu\evidence\claims_matrix.md",
          os.path.join(BASE, "negative_results.md"),
          os.path.join(BASE, "claims_matrix.md")):
    print("  %s  %s" % (md5(p), p))

print()
print("=" * 78)
print("6. THE ARITHMETIC THIS PASS DECLARED — re-run from the artefacts")
print("=" * 78)
import json
runs = {"seed0": "scored1_seed0_test.json",
        "seed1": "scored2_seed1_test.json",
        "seed2": "scored3_seed2_test.json"}
root = r"C:\chlu\evidence\outputs\_thinkpad\w9\w9-maze\runs"
printed = {"budget_infeasible": 0.9601, "walled": 0.9801,
           "mt_unseen": 0.9781, "grounded_possible": 0.8182}
for k, p in printed.items():
    v = [json.load(open(os.path.join(root, f)))["display"]["refusal"][k]["refusal"]
         for f in runs.values()]
    m = sum(v) / 3.0
    print("  %-20s seeds %s  mean %.6f  vs LOG %.4f  match4dp=%s"
          % (k, " / ".join("%.6f" % x for x in v), m, p, round(m, 4) == p))

sep = json.load(open(r"C:\chlu\evidence\outputs\_thinkpad\w9\w9-harness\refusal_separability.json"))
aucs = {k: v["all impossible"]["ridge_auc"] for k, v in sep.items()}
ns   = {k: v["all impossible"]["n_test"] for k, v in sep.items()}
print("  ridge AUC (possible vs all impossible), 6 runs : min %.4f max %.4f ; n_test set %s"
      % (min(aucs.values()), max(aucs.values()), sorted(set(ns.values()))))
print("  runs above 0.50 :", sum(1 for x in aucs.values() if x > 0.50), "of", len(aucs))
bs = {k: v["budget scalar alone"]["auc"] for k, v in sep.items()}
print("  budget scalar alone : min %.4f max %.4f ; runs >= 0.50 : %d"
      % (min(bs.values()), max(bs.values()), sum(1 for x in bs.values() if x >= 0.50)))
