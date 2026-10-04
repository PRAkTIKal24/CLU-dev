"""Compare two tap digests as multisets of plotted-data hashes.
Calls whose every argument is an empty array (legend proxies) are ignored."""
import json, sys
from collections import Counter
def load(p):
    c = json.load(open(p))["calls"]
    return Counter([x for x in c if set(x.split(":")[1].split(",")) != {"EMPTY"}])
a, b = load(sys.argv[1]), load(sys.argv[2])
only_a, only_b = a - b, b - a
print(f"original data calls: {sum(a.values())}   new data calls: {sum(b.values())}")
print(f"in ORIGINAL not in NEW : {sum(only_a.values())}")
for k, v in sorted(only_a.items()): print("   -", k, "x", v)
print(f"in NEW not in ORIGINAL : {sum(only_b.values())}")
for k, v in sorted(only_b.items()): print("   +", k, "x", v)
print("VERDICT:", "IDENTICAL DATA" if not only_a and not only_b else "DATA DIFFERS")
