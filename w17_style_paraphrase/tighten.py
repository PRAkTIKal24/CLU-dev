import re, sys
E = [
('01','A world model trusted with requests it has never seen must satisfy three requirements simultaneously:','A world model trusted with requests it has never seen needs three things at once:'),
('01','Compositional generation is an extensively studied field and exact unlearning an established one;','Compositional generation is extensively studied and exact unlearning established;'),
('01','We further show that, on this object, deletion changes','We show that on this object deletion changes'),
('01','and the zero-step arm is bit-identical to the static read.','its zero-step arm bit-identical to the static read.'),
('01','The generation of held-out factor combinations is a long-standing and active problem, which two lines of work address.','Generating held-out combinations of factors is a long-standing and active problem.'),
('01','For the store, value, conjunction and item deletion are light and data-free, whereas each costs the rivals a full retrain.','Value, conjunction and item deletion are light and data-free for the store and cost the rivals a full retrain.'),
('02','in the orthonormal basis of each well, and the frozen decoder renders the image.',"in each well's orthonormal basis and the frozen decoder renders the image."),
('02','and we assert this on the deployed object','as asserted on the deployed object'),
('02','Labels therefore touch the model only at the address book. This is the \\textbf{interface split}: unsupervised content,','Labels thus touch the model only at the address book: this is the \\textbf{interface split} of unsupervised content,'),
('02','Alongside the content, the store keeps the running covariance, mean and count of the fit rows in each well.','Beside the content, the store keeps per well the running covariance, mean and count of the fit rows.'),
('02','We use a single definition of exactness throughout, and state its scope before any table:','We use one definition of exactness throughout, scoped before any table:'),
('02','so the weights fitted with the deleted rows present remain untouched by any deletion.','so weights fitted with the deleted rows present are untouched by any deletion.'),
('02','We carry the scoping sentence that our deletion theory states','We carry the scoping sentence our deletion theory states'),
('02','tolerance, whereas exact \\emph{deletion}','tolerance; exact \\emph{deletion}'),
('02','On Shapes3D, each rendered image is judged by an external classifier, hash-pinned per seed.','On Shapes3D an external classifier, hash-pinned per seed, judges each rendered image.'),
('02','Every table carries the seed count as a column.','Seed count is a column of every table.'),
('03','This section measures the three properties on Shapes3D, for one deletable object at one partition. The deletable object','This section measures the three properties on Shapes3D. The deletable object'),
('03',"Two caveats accompany the rival rows. First, the ladder's","Two caveats accompany the rival rows: the ladder's"),
('03',"re-fitted on the ladder's rungs. Second, nothing in this table","re-fitted on the ladder's rungs; and nothing in this table"),
('04','novel valid tuples per corner, and judge accuracy at 0.8958, both before and after the deletion.','novel valid tuples per corner before and after, and judge accuracy at 0.8958 before and after.'),
('04','For this reason, refusal needs its own mechanism','This is why refusal needs its own mechanism'),
('04','Dropping the statistic while keeping the atoms and address book yields','Dropping the statistic and keeping the atoms and address book gives'),
('04','which is both the price it does not pay','which is the price it does not pay'),
('04','beyond a margin declared in advance. Total deployed bytes are matched to within','beyond a margin declared in advance, at total deployed bytes matched to within'),
('04','and parameters to within $-0.304$','and parameters within $-0.304$'),
('04','with the arm carrying more parameters','the arm carrying more parameters'),
('05','\\S\\ref{sec:shapes} measures the three properties on images. A route task with an exact integer judge is, however, the harder test of them: one wrong action is a zero. This section therefore measures the same properties on a colour maze','\\S\\ref{sec:shapes} tests the three properties on images; a route task with an exact integer judge is, however, the harder test: one wrong action is a zero. We therefore use a colour maze'),
('05','It is the \\emph{within-maze} floor, and it cannot transfer across mazes:','It is the \\emph{within-maze} floor and cannot transfer across mazes:'),
('05','The single-decode pipeline ceiling bounds an arm that emits one latent and decodes once; it does not bound the two-decode merge.','The single-decode pipeline ceiling, which bounds an arm that emits one latent and decodes once, does not bound the two-decode merge.'),
('05',"On the maze, the store's differentiator is therefore deletion,","The store's differentiator on the maze is therefore deletion,"),
('05','the maze section of this paper consists of','the maze section of this paper is'),
('05','and those negatives; this is still a complete answer','and those negatives, still a complete answer'),
('06','the coverage of very recent preprints and non-indexed workshop papers is incomplete; we judge the risk to be low, not zero.','very recent preprints and non-indexed workshop papers are incompletely covered; we judge the risk low, not zero.'),
('06','with the step count of the settling loop set to zero.',"with the settling loop's step count set to zero."),
('06',"Inside a well this is an algebraic identity. The store's energy is the distance to the atom span, and the request's basin sits at an address inside that span; both minima therefore coincide, and an in-well settle","Inside a well this is an algebraic identity: the store's energy is the distance to the atom span, the request's basin sits at an address inside that span, so both minima coincide and an in-well settle"),
('06','(median displacement 0.39). The dynamics also pay 81,312 to 84,216 bytes of read-side coupling that the static read does not.','(median displacement 0.39), while paying 81,312 to 84,216 bytes of read-side coupling that the static read does not.'),
('06',"The laws of the class are this paper's own theory, each with a status label and a rider","Its laws are this paper's own theory, each with a status label and rider"),
('06',"When switched on, the field store's relaxation","Switched on, the field store's relaxation"),
('06','and it beats a one-parameter form','and beats a one-parameter form'),
('06','on a learned store, with 0 bytes differing','on a learned store, 0 bytes differing'),
('06','since no arm of ours fails the retention law, it distinguishes nothing. Appendix~\\ref{app:physics} gives the full statements, bars and misses.','no arm of ours fails the retention law, so it distinguishes nothing. Full statements, bars and misses: Appendix~\\ref{app:physics}.'),
('06','When switched on, the dynamics cost','Switched on, the dynamics cost'),
('06','The experiment that would turn the law into a claim','The experiment that would make the law a claim'),
]
import glob
for key, old, new in E:
    f = glob.glob('patches/%s-*.tex' % key)[0]
    t = open(f).read()
    pat = re.escape(old)
    pat = re.sub(r'(\\ )+', r'\\s+', pat)  # any whitespace/newline between words
    ms = list(re.finditer(pat, t))
    if len(ms) != 1:
        print('MATCH COUNT %d in %s for: %s' % (len(ms), f, old[:70])); sys.exit(1)
    t = t[:ms[0].start()] + new + t[ms[0].end():]
    open(f, 'w').write(t)
print('applied', len(E))
