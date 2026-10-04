import sys; sys.path.insert(0, '/Users/batmac/clu/outputs/w20-referee3-fixes/tools')
from common_edits import COMMON
P='/Users/batmac/clu/outputs/w17-paper/'
for f in ['DRAFT-v2.tex','DRAFT-v2-style.tex']:
    t=open(P+f).read()
    for k,o,n in COMMON:
        c=t.count(o)
        if c!=1: print(f,k,'count',c)
print('probe done')
