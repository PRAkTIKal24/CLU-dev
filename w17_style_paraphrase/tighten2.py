import re, sys, glob
E = [
('01','The three properties, however, pull against each other:','However, the three pull against each other:'),
('01','The object we study therefore makes the three properties properties of one store,','The object we study therefore makes them properties of one store,'),
('01','We summarise the contributions below, each at the thickness of its evidence.\n',''),
('02',"and the frozen decoder renders the image.","and the frozen decoder the image."),
('02','Labels thus touch the model only at the address book: this is','Labels thus enter only at the address book: this is'),
('02','We carry the scoping sentence our deletion theory states for its own store:',"We carry our deletion theory's scoping sentence for its own store:"),
('04','at every rung and every seed, at 712,072','at every rung and seed, at 712,072'),
('04','The two reads are equal because the statistic is not consulted at generation time.','The statistic is not consulted at generation time, hence the equal reads.'),
('05','the transition and observation instruments, whose rows are pending','transition and observation instruments with rows pending'),
('05','Consequently, no row is claimed for the store under either verb.','No row is therefore claimed for the store, on either verb.'),
('05','is part of the \\emph{interface}, and it is task-shaped: images compose by mixing latents, whereas routes compose by overlaying constraints.','is part of the \\emph{interface} and is task-shaped: images compose by mixing latents; routes, by overlaying constraints.'),
('05',"The store's differentiator on the maze is therefore deletion, which transferred (Table~\\ref{tab:maze-del}); the composition result on routes belongs to the verb.","The store's differentiator on the maze is deletion, which transferred (Table~\\ref{tab:maze-del}); composition on routes belongs to the verb."),
]
for key, old, new in E:
    f = glob.glob('patches/%s-*.tex' % key)[0]
    t = open(f).read()
    if old.endswith('\n'):
        n = t.count(old); 
        if n != 1: print('COUNT', n, old[:60]); sys.exit(1)
        t = t.replace(old, new); open(f,'w').write(t); continue
    pat = re.sub(r'(\\ )+', r'\\s+', re.escape(old))
    ms = list(re.finditer(pat, t))
    if len(ms) != 1: print('MATCH COUNT %d in %s: %s' % (len(ms), f, old[:70])); sys.exit(1)
    t = t[:ms[0].start()] + new + t[ms[0].end():]; open(f,'w').write(t)
print('applied', len(E))
