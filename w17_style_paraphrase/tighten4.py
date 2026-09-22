import re, sys, glob
E = [
('06','1.0000 on the seen row); no row is therefore claimed for the store (Appendix~\\ref{app:cmaze2}). On CMAZE-2b the controls again fail to collapse','1.0000 on the seen row), so no row is claimed for the store (Appendix~\\ref{app:cmaze2}). On CMAZE-2b they again fail to collapse'),
('06','close to solving the maze, which is a property of the task rather than a defect of the model.','close to solving the maze, a property of the task rather than a defect of the model.'),
('06','The Shapes3D coupling hole is a different object in a different environment and is unaffected.','The Shapes3D coupling hole, a different object in a different environment, is unaffected.'),
('06','misses on all three seeds, and the law predicts where its simplest form stops.','misses on all three seeds; the law predicts where its simplest form stops.'),
('06','The stronger claim, that the post-delete store equals the never-inserted one, is','The stronger claim that the post-delete store equals the never-inserted one is'),
]
for key, old, new in E:
    f = glob.glob('patches/%s-*.tex' % key)[0]
    t = open(f).read()
    pat = re.sub(r'(\\ )+', r'\\s+', re.escape(old))
    ms = list(re.finditer(pat, t))
    if len(ms) != 1: print('MATCH COUNT %d in %s: %s' % (len(ms), f, old[:70])); sys.exit(1)
    t = t[:ms[0].start()] + new + t[ms[0].end():]; open(f,'w').write(t)
print('applied', len(E))
