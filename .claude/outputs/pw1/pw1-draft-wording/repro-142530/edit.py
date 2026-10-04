#!/usr/bin/env python3
"""PW-1 wording items 7, 8 and 13 in one draft (DRAFT-v2.tex or its style twin).

usage: edit.py IN.tex OUT.tex REPORT.json

Every operation is a list of alternatives (old, new). `old` is literal text, matched
whitespace-tolerantly; the first alternative whose anchor occurs exactly once is applied, and an
operation with no applicable alternative fails loudly. The two drafts differ in prose (the style
twin is a sentence-by-sentence paraphrase), so a site whose wording differs carries one alternative
per draft; every `new` that carries a number is the same in both drafts, so the style diff-check on
numbers stays empty. Numbers that leave the text keep a pointer in a
`% pw1-draft-wording (item N): removed ...; its numbers stay in: ...` comment (the provenance keyword
is avoided there, as in pw1_draft_cuts and pw1_draft_figures).
"""
import json, re, sys

IN, OUT, REP = sys.argv[1:4]
text = open(IN).read()
report = []


def pat(lit):
    parts = re.split(r'(\s+)', lit)
    return ''.join(r'\s+' if p.strip() == '' and p else re.escape(p) for p in parts)


def code_only(s):
    return '\n'.join(re.split(r'(?<!\\)%', l, 1)[0] for l in s.split('\n'))


def nums(s):
    c = re.sub(r'\\(ref|label)\{[^}]*\}', ' ', code_only(s))
    c = re.sub(r'\\sci\{([^}]*)\}\{([^}]*)\}', r' \1e\2 ', c)
    return [n.rstrip('.,') for n in re.findall(r'(?<![A-Za-z_\\{])(\d[\d,]*\.?\d*(?:e-?\d+)?)', c)]


def op(item, what, *alts, ptr=None):
    global text
    for k, (old, new) in enumerate(alts):
        ms = list(re.finditer(pat(old), text, re.S))
        if len(ms) == 1:
            break
    else:
        raise SystemExit('%s: no alternative anchors exactly once (counts %s)' % (
            what, [len(re.findall(pat(o), text, re.S)) for o, _ in alts]))
    m = ms[0]
    was = m.group(0)
    left = [n for n in nums(was) if n not in nums(new)]
    added = [n for n in nums(new) if n not in nums(was)]
    assert ('{PTR}' in new) == bool(ptr), what
    new = new.replace('{PTR}', '%% pw1-draft-wording (item %s): removed %s; its numbers stay in: %s\n'
                      % (item, what, '; '.join(ptr or [])))
    line = text.count('\n', 0, m.start()) + 1
    text = text[:m.start()] + new + text[m.end():]
    report.append(dict(item=item, what=what, alternative=k, line_before=line, before=was, after=new,
                       numbers_leaving=left, numbers_added=added))


# ------------------------------------------------------------------ item 13: the title
op(13, 'title',
   (r'\title{naive-CLU: Exactly Deletable Compositional Generation\\ on a Frozen Encoder}',
    r'\title{naive-CLU: Deletable Compositional Generation\\ on a Frozen Encoder}'))

# ------------------------------------------------------------------ item 13: abstract
op(13, 'abstract: forget rows exactly; exact unlearning',
   (r'''\emph{forget} specific training rows exactly. Compositional generation is a crowded area
and exact unlearning is an established one.''',
    r'''\emph{forget} specific training rows. Compositional generation is a crowded area
and machine unlearning is an established one.'''),
   (r'''\emph{forget} specific
training rows exactly. Compositional generation is extensively studied and exact unlearning established.''',
    r'''\emph{forget} specific
training rows. Compositional generation is extensively studied and machine unlearning established.'''))

op(13, 'abstract: the deletion sentences (plain phrasing, scope, limits; no general post-deletion-composition claim)',
   (r'''Deletion is exact on the n-CLU and on its address book: after a value, a conjunction or a random
1\% of items is deleted, both equal the objects refit without those rows (content mode deviation at
most \sci{5.38}{-07}, book statistic deviation 0). It is not free: the book's exact statistic is % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
0.97 GB and its refit takes 2,367 to 5,913 s, and the deletable object is 4.0768 to 4.0897 times % SRC: w20-shapes-data-matched/FINDINGS.md; w13-seeds-and-tables/TABLES.md; w13-referee-2/REFEREE-v1.md
the 524,288-byte reference in its footprint breakdown. Value and item deletions leave composition on % SRC: w13-referee-2/REFEREE-v1.md; w13-seeds-and-tables/TABLES.md
the surviving combinations within one tuple per held-out combination, but deleting the conjunction red $\wedge$ sphere
changes what the book knows about the shared value: survivors fall by 6.13 / 6.43 / 6.30 tuples per % SRC: w20-shapes-data-matched/FINDINGS.md
combination, against at most 0.11 for a random 1\% deletion. % SRC: w20-shapes-data-matched/FINDINGS.md''',
    r'''After a value, a conjunction or a random 1\% of items is deleted, the n-CLU matches an n-CLU that
was never fitted on the deleted rows, and its address book likewise (content mode deviation at
most \sci{5.38}{-07}, book statistic deviation 0); the frozen encoder and decoder are outside the
object. Deletion has limits, which we print: the book's retained statistic is % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
0.97 GB and its refit takes 2,367 to 5,913 s; the deletable object is 4.0768 to 4.0897 times % SRC: w20-shapes-data-matched/FINDINGS.md; w13-seeds-and-tables/TABLES.md; w13-referee-2/REFEREE-v1.md
the 524,288-byte reference in its footprint breakdown; and deleting the conjunction red sphere % SRC: w13-referee-2/REFEREE-v1.md; w13-seeds-and-tables/TABLES.md
changes what the book knows about the shared value: survivors fall by 6.13 / 6.43 / 6.30 tuples per % SRC: w20-shapes-data-matched/FINDINGS.md
combination, against at most 0.11 for a random 1\% deletion. % SRC: w20-shapes-data-matched/FINDINGS.md'''))

# ------------------------------------------------------------------ item 13: introduction / positioning
op(13, 'positioning: exact unlearning is established',
   (r'Exact unlearning is also established, for $k$-means',
    r'Unlearning that matches a retrain without the deleted rows is also established, for $k$-means'),
   (r'Exact unlearning is likewise established: for $k$-means',
    r'Unlearning that matches a retrain without the deleted rows is likewise established: for $k$-means'))
op(13, 'screen: none exact',
   (r'false positives and none exact;', r'false positives and none deleting training rows;'))
op(13, 'screen: the orthogonal query',
   (r'''\emph{compositional} $\wedge$
\emph{generative} $\wedge$ \emph{exact deletion}, returns zero works''',
    r'''\emph{compositional} $\wedge$
\emph{generative} $\wedge$ a deletion phrase, returns zero works'''))
op(13, 'SISA: exactly unlearnable',
   (r'exactly unlearnable by splitting its data', r'unlearnable in this sense by splitting its data'))
op(13, 'SISA: only exact route',
   (r'whose only exact route is the retrain of', r'whose only route to that match is the retrain of'))
op(13, 'decremental PCA: our route is exact',
   (r'Our route is exact for the opposite reason:', r'Our route matches the refit, for the opposite reason:'),
   (r'''Our route is exact for the opposite
reason:''', r'''Our route matches the refit, for the opposite
reason:'''))
op(13, 'Sec. why: the deletion property defined plainly',
   (r'''\textbf{Exact deletion}: after a set of training rows
is deleted, the deletable object equals, to floating-point roundoff, the object fitted without
those rows.''',
    r'''\textbf{Deletion}: after a set of training rows
is deleted, the deletable object matches, to floating-point roundoff, an object that was never fitted
on those rows.'''),
   (r'''\textbf{Exact deletion}: after a set of training rows is deleted, the deletable object
  equals, to floating-point roundoff, the object fitted without those rows.''',
    r'''\textbf{Deletion}: after a set of training rows is deleted, the deletable object
  matches, to floating-point roundoff, an object that was never fitted on those rows.'''))
op(13, 'Sec. why: lookup table deletes exactly',
   (r'a lookup table deletes exactly but', r'a lookup table deletes in this sense but'))
op(13, 'Sec. why: can delete exactly only by retraining',
   (r'composes but can delete exactly only by', r'composes but meets this definition only by'))
op(13, 'Sec. why: approximate by construction, not exact',
   (r'is approximate by construction, not exact (Sec.~\ref{sec:del-comparators}).',
    r'is approximate by construction and does not meet it (Sec.~\ref{sec:del-comparators}).'))

op(13, 'contribution 2: deletion, plain phrasing and scope, limitations, no general post-deletion-composition claim',
   (r'''\item \textbf{Exact deletion on the n-CLU and on its address book, with its cost and its limitation
  printed.} Value, conjunction and item deletion leave the n-CLU and the refit address book equal to
  those fitted without the rows. Value and item deletion move composition on the surviving combinations by
  at most 0.61 tuples per held-out combination, but deleting a conjunction measurably changes what the book knows % SRC: w20-shapes-data-matched/FINDINGS.md
  about the value it shares with the survivors (Sec.~\ref{sec:shapes-del}). The deletion reads no
  training row, but the book's exact statistic is 0.97 GB, the size of its fit data, and its refit % SRC: w20-shapes-data-matched/FINDINGS.md''',
    r'''{PTR}\item \textbf{Deletion on the n-CLU and on its address book, with its limitations printed.} After
  a value, conjunction or item deletion, the n-CLU matches an n-CLU that was never fitted on the deleted
  rows, and the refit address book matches a book never fitted on them; the match is on the n-CLU and
  its address book, and the frozen encoder and decoder are outside the object. We make no general claim
  about composition after deletion: deleting a conjunction measurably changes what the book knows % SRC: w20-shapes-data-matched/FINDINGS.md
  about the value it shares with the survivors (Sec.~\ref{sec:shapes-del}). The deletion reads no
  training row, but the book's retained statistic is 0.97 GB, the size of its fit data, and its refit % SRC: w20-shapes-data-matched/FINDINGS.md'''),
   ptr=['w20-shapes-data-matched/FINDINGS.md (0.61: the value deletion\'s largest survivor move, also in Sec. shapes-del)'])

# ------------------------------------------------------------------ item 13: object section
op(13, 'object: paragraph title',
   (r'\paragraph{The retained statistic and exact deletion.}', r'\paragraph{The retained statistic and deletion.}'))
op(13, 'object: the book\'s own exact statistic',
   (r'The address book keeps its own exact statistic,', r'The address book keeps its own retained statistic,'))
op(13, 'object: one definition of exactness',
   (r'We use one definition of exactness throughout, and state its scope',
    r'We use one definition of deletion throughout, and state its scope'))
op(13, 'object: the scoping sentence',
   (r'''\textsc{Exact is scoped to the n-CLU and its address book.} After deleting a set of rows, the n-CLU's
content and the address book equal, to float roundoff, those fitted from the retained rows alone,
holding the frozen encoder and decoder fixed. Neither frozen network is refit at any point.''',
    r'''\textsc{The match is on the n-CLU and its address book; the frozen encoder and decoder are outside
the object.} After deleting a set of rows, the n-CLU matches an n-CLU that was never fitted on the
deleted rows, and the address book likewise: content and book equal, to float roundoff, those fitted
from the retained rows alone. Neither frozen network is refit at any point.'''))

# ------------------------------------------------------------------ item 13: Sec. 3.3 / 3.4 / 3.5 / 3.6
op(13, 'Sec. 3.3 title',
   (r'\subsection{Exact deletion on the single design: value, conjunction, item}',
    r'\subsection{Deletion on the single design: value, conjunction, item}'))
op(13, 'Sec. 3.3: reads its exact statistic',
   (r'reads its exact statistic, 967,304,800 B', r'reads its retained statistic, 967,304,800 B'))
op(13, 'Sec. 3.3: the probe (exactness is n-CLU-scoped)',
   (r'''\paragraph{The probe that bounds the claim.} Exactness is n-CLU-scoped in a way that is
measurable, not only definitional.''',
    r'''\paragraph{The probe that bounds the claim.} That the match holds on the n-CLU and its address book,
and not beyond, is measurable, not only definitional.'''),
   (r'''\paragraph{The probe that bounds the claim.} The n-CLU-scoping of exactness is measurable, not
only definitional.''',
    r'''\paragraph{The probe that bounds the claim.} That the match holds on the n-CLU and its address book,
and not beyond, is measurable, not only definitional.'''))
op(13, 'Sec. 3.4: only exact deletion route',
   (r"A learned baseline's only exact deletion route is to retrain from scratch without the rows.",
    r"A learned baseline's only route to a model never fitted on the rows is to retrain from scratch without them."))
op(13, "Sec. 3.4: the n-CLU's exact deletion",
   (r"cleaner than the n-CLU's exact deletion.", r"cleaner than the n-CLU's deletion."))
op(13, 'Sec. 3.4: exactness by construction',
   (r'is exactness by construction and no pool row read.',
    r'is the match with a refit without the rows, by construction, and no pool row read.'))
op(13, 'Sec. 3.5: a second exact mechanism',
   (r'Refusal is a second exact mechanism on the same object', r'Refusal is a second mechanism on the same object'))
op(13, 'Sec. 3.5: exact on bytes and free on behaviour',
   (r'''The operation that is both exact on bytes and free on behaviour is the hole, not
deletion in general.''',
    r'''The operation that rewrites only the bytes it names and leaves behaviour unchanged is the
hole, not deletion in general.'''),
   (r'''The hole, not deletion in general, is the operation that is both exact on bytes and
free on behaviour.''',
    r'''The hole, not deletion in general, is the operation that rewrites only the bytes it names and
leaves behaviour unchanged.'''))
op(13, 'Sec. 3.5: changes it by exactly +0.0000',
   (r'changes it by exactly $+0.0000$.', r'changes it by $+0.0000$.'),
   (r'with a change of exactly $+0.0000$.', r'with a change of $+0.0000$.'))
op(13, 'Sec. 3.5: the claim is exactness',
   (r'The claim is exactness, never erasure without loss.',
    r'The claim is a match with a refit without the rows, never erasure without loss.'))
op(13, 'Sec. 3.6: has no exact deletion',
   (r'the reason it has no exact deletion.', r'the reason it cannot delete in this sense.'))

# ------------------------------------------------------------------ item 8: CMAZE-T
op(8, 'CMAZE-T lead-in: seeds differ sharply in difficulty',
   (r'its seeds differ sharply in difficulty,', r'its results differ widely from seed to seed,'))

# ------------------------------------------------------------------ item 7: the seed-5 limitation
op(7, 'CMAZE-T: the seed-5 miss (content-capacity failure of the 6-atom design; not a hold-out effect)',
   (r'''held-out tuples that are fully supported in training and that eight other arms solve. Every landscape
control stays below the n-CLU on every seed (highest 0.495, seed 5).''',
    r'''held-out tuples that are fully supported in training and that eight other arms solve. This miss is
not an effect of the hold-out draw: both tuples share one outcome code (legal, move south, pocket 2),
and the miss is a content-capacity failure of the 6-atom design on that code; the 10-atom relational
design (n-CLU-REL, below) scores every row of both tuples valid. % pw1-draft-wording (item 7): mechanism as ruled in PW-1 PLAN item 7 (the W23 diagnosis); the code's 0.04 decoding margin and its 106 of 317 wrong fit rows are in no record on this machine and are not printed. Checked: tuple keys "2,0,1,2" / "2,0,3,2" (pocket, door, tint, direction; ACTIONS "NESWP" in chlu/exp_w15_cmaze_t.py, so direction 2 = south, and no door with pocket 2 and tint 1 or 3 is legal) in w21-maze-t-seed4/replay/check_BOTH_test_seed5_PERTUPLE.json PER_TUPLE_NWAY.CLU = [90, 0] / [80, 0]; n-CLU-REL w21-rel-newseeds/replay/BOTH_test_seed5_PERTUPLE.json PER_TUPLE_NWAY.CLU = [90, 90] / [80, 80]; 10 atoms: w21-rel-newseeds/config/CLU-REL-A10.json n_atoms; "not a hard-seed effect": w21-maze-t-seed4/FINDINGS.md
Every landscape
control stays below the n-CLU on every seed (highest 0.495, seed 5).'''))

# ------------------------------------------------------------------ item 7: relational coding's limit; "16 of 16" SRC
op(7, 'n-CLU-REL paragraph: SRC block names figures-redraw2 (the seed-5 tie, 16/16) and the value-row table',
   (r'''% SRCBLOCK: w21-rel-newseeds/FINDINGS.md; w21-maze-t-clu-improve/FINDINGS.md; w21-maze-t-seed4/FINDINGS.md
\paragraph{A relational request coding for the n-CLU is an input-representation result.}''',
    r'''% SRCBLOCK: w21-rel-newseeds/FINDINGS.md; w21-maze-t-clu-improve/FINDINGS.md; w21-maze-t-seed4/FINDINGS.md; ../../outputs/w22-figures-redraw2/FINDINGS.md
\paragraph{A relational request coding for the n-CLU is an input-representation result.}'''))
op(7, 'n-CLU-REL paragraph: the relational-coding limit (value row)',
   (r'''relational baselines is sizeable only on seed 2 (+0.165). We report the configuration as evidence about
request coding, which any learner can be given.''',
    r'''relational baselines is sizeable only on seed 2 (+0.165). Relational coding has one limit: relations
reach the n-CLU, colour identity does not. On the never-trained value the never-seen colour's
relations fire normally, but every column that names the colour is zero, so the value row stays a
loss: n-CLU-REL reads 0.554 against the best matched baseline's 0.797 (six-seed means) and is % SRC: w21-rel-newseeds/TABLES.md
behind on all six seeds. We report the configuration as evidence about
request coding, which any learner can be given.'''))

# ------------------------------------------------------------------ item 13: CMAZE-T deletion sentence
op(13, 'CMAZE-T: exact deletion holds',
   (r'''Exact deletion holds on
words plus observation on every seed: the downdated statistic matches the statistic fitted without
the rows to a relative \sci{1.94}{-15} at worst, the deployed atoms do not change, and the downdated
n-CLU reads exactly as the n-CLU fitted without the rows,''',
    r'''On words plus
observation, on every seed, the n-CLU after deletion matches an n-CLU that was never fitted on the
deleted rows: the downdated statistic matches the statistic fitted without the rows to a relative
\sci{1.94}{-15} at worst, the deployed atoms do not change, and the downdated n-CLU reads identically
to the n-CLU fitted without the rows,'''))

# ------------------------------------------------------------------ item 13: limitations
op(13, 'limitation 2: deletion is exact on the n-CLU',
   (r'''Deletion
  is exact on the n-CLU and on the address book, but deleting the red sphere lowers''',
    r'''After deletion
  the n-CLU and the address book match those never fitted on the deleted rows, but deleting the red
  sphere lowers'''))
op(13, 'limitation 3: exact deletion is not cheap; the book\'s exact statistic',
   (r"\item \textbf{Exact deletion is not cheap in memory or time.} The book's exact statistic is 0.97 GB,",
    r"\item \textbf{Deletion is not cheap in memory or time: the address book's refit.} The book's retained statistic is 0.97 GB,"))
op(13, 'limitation 4: exact requires a large number of bytes',
   (r'\item \textbf{Exact requires a large number of bytes.}', r'\item \textbf{Deletability requires a large number of bytes.}'))

# ------------------------------------------------------------------ item 13: leaving note
op(13, 'leaving note: provides exact deletion by subtraction',
   (r'The naive unit provides exact deletion by subtraction and,',
    r'The naive unit provides deletion by subtraction, matching an n-CLU never fitted on the deleted rows, and,'))
op(13, 'leaving note: the cost of exactness',
   (r"memory and time cost of exactness, in bytes and in the book's refit.",
    r"memory and time cost of that match, in bytes and in the book's refit."))

# ------------------------------------------------------------------ item 13: future work (the empirical comparison)
FUT = r'''

\paragraph{Deletion.} Improving deletion is also future work, and the measured comparison is
the clue: after deletion the n-CLU composes above every baseline retrained without the same rows on
the value and item scopes, and level with the MLP and the LSTM on the red sphere, where its 95\% % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
interval against them includes zero on some seeds (Sec.~\ref{sec:del-comparators}). A deletion that
removed the conjunction's rows without removing what the address book knows about the shared value
would close that gap.'''
op(13, 'follow-up: future-work paragraph on deletion',
   (r"What it must contain is fixed by this study's negatives (App.~\ref{app:bench}).",
    r"What it must contain is fixed by this study's negatives (App.~\ref{app:bench})." + FUT),
   (r"This study's negatives fix what it must contain (App.~\ref{app:bench}).",
    r"This study's negatives fix what it must contain (App.~\ref{app:bench})." + FUT))

# ------------------------------------------------------------------ item 13: ethics
ETH = r'''Deletion is relevant to data-protection
requests; we claim only that the n-CLU and its address book match those never fitted on the deleted
rows (the frozen encoder and decoder are outside the object) and no certified unlearning, and
we state in Sec.~\ref{sec:shapes-del} a case in which deleting a conjunction changes what the
model answers about the values that survive it.'''
op(13, 'ethics statement',
   (r'''Exact deletion is relevant to data-protection
requests; we claim exactness on the n-CLU and its address book only and no certified unlearning, and
we state in Sec.~\ref{sec:shapes-del} a case in which exact deletion of a conjunction changes what the
model answers about the values that survive it.''', ETH))

# ------------------------------------------------------------------ item 7: every number verified / traced, scoped
REPRO = r'''Each number in this paper is read from the recorded result of a declared run, and nearly every
one carries its provenance beside it in the source. The verbatim re-check against the cited records
is partial: an automated check covers every number of three groups of runs and, overall, 1,308 of % SRC: ../../outputs/w23-fold/FINDINGS.md
the 2,293 sourced number tokens of an earlier version of this draft, and it checks that a number is % SRC: ../../outputs/w23-fold/FINDINGS.md
present in its record, not what the number means.'''
op(7, 'reproducibility statement: scoped',
   (r'''Every number in this paper is read from the recorded result of a declared run;
the provenance of each is kept beside it in the source.''', REPRO),
   (r'''Every number in this paper is read from the recorded result of a declared run,
and the provenance of each is kept beside it in the source.''', REPRO))
PROV = r'''\paragraph{Provenance.} Nearly every number in the source of this paper carries a comment naming
the record it was read from; the reproducibility statement gives the scope of the verbatim re-check.'''
op(7, 'App. conventions: provenance paragraph, scoped',
   (r'''\paragraph{Provenance.} Every number in the source of this paper carries a comment naming the
record it was read from.''', PROV),
   (r'''\paragraph{Provenance.} In the source of this paper, every number carries a comment naming the
record from which it was read.''', PROV))

# ------------------------------------------------------------------ item 13: appendices
op(13, 'Tab. impl caption',
   (r'\caption{\textbf{Exact (n-CLU-scoped) deletion on Shapes3D, implementations 1 and 2, bands not',
    r'\caption{\textbf{Deletion on Shapes3D, matched on the n-CLU, implementations 1 and 2, bands not'))
op(13, 'App. impl: outcome for an exactness claim',
   (r'the best available outcome for an exactness claim.',
    r'the best available outcome for a claim that the downdate matches the refit.'),
   (r'''the best available outcome
for an exactness claim.''', r'''the best available outcome
for a claim that the downdate matches the refit.'''))
op(13, 'App. impl: the exactness assertion failed',
   (r'The exactness assertion failed first,', r'The assertion that the downdate matches the refit failed first,'),
   (r'The exactness assertion first failed', r'The assertion that the downdate matches the refit first failed'))
op(13, 'App. impl: the exactness claim removes content',
   (r'the only place the exactness claim removes content.', r'the only place the deletion claim removes content.'),
   (r'the only place where the exactness claim removes content.', r'the only place where the deletion claim removes content.'))
for i in range(3):
    op(13, 'App. impl, K=5 (ward) smoke: exactness value (%d of 3)' % (i + 1),
       (r'rows, exactness \sci{' + ['1.455}{-11}, survivor $\\Delta$ $+0.1250$',
                                    '2.842}{-14}, survivor $\\Delta$ $-0.0833$',
                                    '1.455}{-11}, survivor $\\Delta$\n$+0.0000$'][i],
        r'rows, deviation from the refit \sci{' + ['1.455}{-11}, survivor $\\Delta$ $+0.1250$',
                                                   '2.842}{-14}, survivor $\\Delta$ $-0.0833$',
                                                   '1.455}{-11}, survivor $\\Delta$\n$+0.0000$'][i]))
op(13, 'App. physics: memory-level exact deletion (title and sentence)',
   (r'''\paragraph{Memory-level exact deletion in a designed memory (\textsc{confirmed [prior
derivation]}).} With an order-independent waitlist, memory-level deletion is exact at every load
measured''',
    r'''\paragraph{Memory-level deletion in a designed memory (\textsc{confirmed [prior
derivation]}).} With an order-independent waitlist, memory-level deletion leaves a memory
byte-equal to one never written with the deleted items, at every load measured'''))
op(13, 'Tab. cmazet: deletion, value row',
   (r'colour 1; n-CLU-scoped exactness $S_k$ relative change',
    r'colour 1; the n-CLU after deletion matches the n-CLU fitted without the rows: $S_k$ relative change'))
op(13, 'Tab. cmazet-sym: deletion row',
   (r'''base n-CLU, words + observation: n-CLU-scoped exactness on every seed (relative statistic change at most \sci{1.94}{-15}, atoms unchanged, reads after the downdate identical to the n-CLU fitted without the rows);''',
    r'''base n-CLU, words + observation: on every seed the n-CLU after deletion matches the n-CLU fitted without the rows (relative statistic change at most \sci{1.94}{-15}, atoms unchanged, reads after the downdate identical);'''))

# ------------------------------------------------------------------ item 7: the two SRC fixes on the removal pointers
op(7, 'SRC fix: 0.6407 (pointer of the removed fig:maze-wm caption names the file that prints it)',
   (r'''% pw1-draft-figures (item 1): removed the fig:maze-wm caption (rule text, ceiling, hard-seed split, per-seed relational values), rewritten; its numbers stay in: w21-maze-t-seed4/FINDINGS.md; w21-rel-newseeds/FINDINGS.md; w21-maze-t-seed4/TABLES.md''',
    r'''% pw1-draft-figures (item 1): removed the fig:maze-wm caption (rule text, ceiling, hard-seed split, per-seed relational values), rewritten; its numbers stay in: w21-maze-t-seed4/FINDINGS.md; w21-rel-newseeds/FINDINGS.md; w21-maze-t-seed4/TABLES.md
% pw1-draft-wording (item 7): 0.6407 is printed verbatim in w21-maze-t-seed4/TABLES.md l.11 ("cut = 0.6407"); it rounds the JSON leaf w21-maze-t-seed4/MAZE-T-SEED4.json classification.cut.cut = 0.6407297594090047'''))
op(7, 'SRC fix: 16 of 16 (pointer of the removed App. with Fig. maze-rel-tuples names figures-redraw2)',
   (r'''w21-rel-newseeds/FINDINGS.md; w21-rel-newseeds/test/BOTH_test_seed5.json; w15-maze-deletion-2b/TABLE.md;''',
    r'''w21-rel-newseeds/FINDINGS.md; ../../outputs/w22-figures-redraw2/FINDINGS.md (the relational-baseline half of "16 of 16": "also equals it on 16/16"); w21-rel-newseeds/test/BOTH_test_seed5.json; w15-maze-deletion-2b/TABLE.md;'''))
op(7, 'SRC fix: Tab. cmazet-sym n-CLU-REL row block names figures-redraw2 (the seed-5 tie)',
   (r'''% SRCBLOCK: w21-rel-newseeds/FINDINGS.md; w21-maze-t-seed4/FINDINGS.md
\textbf{words + observation, n-CLU-REL (TEST)}''',
    r'''% SRCBLOCK: w21-rel-newseeds/FINDINGS.md; w21-maze-t-seed4/FINDINGS.md; ../../outputs/w22-figures-redraw2/FINDINGS.md
\textbf{words + observation, n-CLU-REL (TEST)}'''))

open(OUT, 'w').write(text)
json.dump(report, open(REP, 'w'), indent=1)
print('%s: %d operations applied' % (IN.split('/')[-1], len(report)))
