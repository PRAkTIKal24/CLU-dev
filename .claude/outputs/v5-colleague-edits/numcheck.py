# -*- coding: utf-8 -*-
import io, re
B = io.open("/tmp/v5edit/pj_sub_BEFORE.tex", encoding="utf-8").read().split("\n")
A = io.open("/tmp/v5edit/pj_sub.tex", encoding="utf-8").read().split("\n")

def numbers(text):
    # numeric literals, incl. decimals
    return re.findall(r"\d+\.\d+|\d+", text)

# main text = lines 27..133 (0-indexed 26..132) in BEFORE; in AFTER main text is same span (no lines added before 134)
bmain = "\n".join(B[26:133])
amain = "\n".join(A[26:133])
bn = numbers(bmain); an = numbers(amain)
from collections import Counter
bc, ac = Counter(bn), Counter(an)
removed = bc - ac
added = ac - bc
print("NUMBERS REMOVED FROM MAIN TEXT:", sorted(removed.elements()))
print("NUMBERS ADDED TO MAIN TEXT   :", sorted(added.elements()))

# new appendix block = AFTER lines 135..169 (1-indexed) -> 134:169
tbl = "\n".join(A[134:169])
tn = Counter(numbers(tbl))
missing = [x for x in removed.elements() if x not in tn]
print("ORPHANS (left main text, not in new table):", missing)

# every number in the new table must have an ancestor in BEFORE file
before_all = "\n".join(B)
ban = Counter(numbers(before_all))
noanc = [x for x in tn.elements() if x not in ban]
print("TABLE NUMBERS WITHOUT ANCESTOR IN PRE-PASS FILE:", noanc)
