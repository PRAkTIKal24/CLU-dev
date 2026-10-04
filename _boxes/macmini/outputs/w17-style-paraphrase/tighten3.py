import re, sys, glob
E = [
('01','Because deletion downdates the statistic rather than the truncated factors, the store after deletion equals the store fitted without the deleted rows.','Deletion downdates the statistic, not the truncated factors, so the store after deletion equals the store fitted without those rows.'),
('01','Deletion is exact and store-scoped, but it is not free:','Deletion is exact and store-scoped, but not free:'),
('01','but it lifts a matched-bytes lookup floor further, which places the composition operator in the interface.','but it lifts a matched-bytes lookup floor more, so the composition operator belongs to the interface.'),
('06','This section turns from what the naive unit does to what it does not: its measured limitations, the one measured clue toward a dynamical successor, the theory we carry, and a leaving note.','We now state what the naive unit does not do, the one measured clue, and the theory we carry.'),
('06',"The naive unit is the zero-dynamics limit of a settling unit (\\S\\ref{sec:clue}), and the leaving note states what it does not buy. The follow-up is therefore a full evaluation","The leaving note states what the naive unit does not buy; the follow-up is therefore a full evaluation"),
('06',"with the same judges, ledger and matching, and with this paper's store as its zero-dynamics ablation. The negatives of this study fix what it must contain","with the same judges, ledger and matching and this paper's store as its zero-dynamics ablation. This study's negatives fix what it must contain"),
]
for key, old, new in E:
    f = glob.glob('patches/%s-*.tex' % key)[0]
    t = open(f).read()
    pat = re.sub(r'(\\ )+', r'\\s+', re.escape(old))
    ms = list(re.finditer(pat, t))
    if len(ms) != 1: print('MATCH COUNT %d in %s: %s' % (len(ms), f, old[:70])); sys.exit(1)
    t = t[:ms[0].start()] + new + t[ms[0].end():]; open(f,'w').write(t)
print('applied', len(E))
