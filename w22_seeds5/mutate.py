"""Negative-test helper (run inside a scratch copy of the register).
  mutate.py FILE OLD NEW            replace the first OLD by NEW; exit 1 if OLD is absent
  mutate.py --append FILE LINE      append LINE to FILE
  mutate.py --pass2-tamper MANIFEST change the value of one HISTORICAL (w16 report) cell in the pass-2 manifest"""
import json
import sys

a = sys.argv[1:]
if a[0] == "--append":
    with open(a[1], "a", encoding="utf-8") as fh:
        fh.write("\n" + a[2] + "\n")
elif a[0] == "--pass2-tamper":
    j = json.load(open(a[1], encoding="utf-8"))
    k = sorted(json.load(open("TABLES-w18.json", encoding="utf-8"))["cells_historical"])[0]  # a cell the build DOES read from the manifest
    j["cells"][k]["value"] += "X"
    json.dump(j, open(a[1], "w", encoding="utf-8"))
else:
    f, old, new = a
    s = open(f, encoding="utf-8").read()
    if old not in s:
        sys.exit(f"anchor not found in {f}: {old!r}")
    open(f, "w", encoding="utf-8").write(s.replace(old, new, 1))
