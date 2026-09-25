#!/usr/bin/env python3
"""PW-1 items 14, 15 and 17: the maze section is CMAZE-T only; Fig. maze-rel-tuples and Fig. cmazet-deletion out.

usage: cuts.py IN.tex OUT.tex REPORT.json {v2|style}

Every operation is anchored on literal text and asserts that its anchor occurs exactly once, so a
draft that has drifted fails loudly instead of being cut in the wrong place. Operations marked for one
draft only exist where the style twin paraphrases the original. Removed numbers keep a pointer in a
`% pw1-draft-cuts (item N)` comment naming the records they stay in (the removed block's own provenance
paths). The pointer comments deliberately avoid the provenance keyword so the style diff-check's
provenance multisets compare only the surviving provenance comments.
"""
import json, re, sys

IN, OUT, REP, WHICH = sys.argv[1:5]
text = open(IN).read()
orig = text
report = []
SEP = '% ' + '-' * 87 + '\n'
BAR = '% ' + '=' * 87 + '\n'


def lineno(t, pos):
    return t.count('\n', 0, pos) + 1


def code_only(s):
    return '\n'.join(re.split(r'(?<!\\)%', l, 1)[0] for l in s.split('\n'))


def words(s):
    """Rendered-word estimate: comments stripped, control words and braces dropped."""
    c = code_only(s)
    c = re.sub(r'\\(label|ref|eqref|cite[pt]?|nocite|begin|end|setlength|figslot)\{[^}]*\}', ' ', c)
    c = re.sub(r'\\[A-Za-z]+\*?', ' ', c)
    c = re.sub(r'[{}$&\\~^_]', ' ', c)
    return len(re.findall(r'[A-Za-z0-9][A-Za-z0-9\'.,/:+\-]*', c))


def srcs(s):
    out = []
    for l in s.split('\n'):
        m = re.search(r'SRC(?:BLOCK)?:\s*(.*)', l)
        if m:
            for p in m.group(1).split(';'):
                p = p.strip()
                if p and p not in out:
                    out.append(p)
    return out


def refs(s):
    return re.findall(r'\\ref\{([^}]*)\}', code_only(s))


def labels(s):
    return re.findall(r'\\label\{([^}]*)\}', code_only(s))


def nums(s):
    c = re.sub(r'\\(ref|label)\{[^}]*\}', ' ', code_only(s))
    return re.findall(r'(?<![A-Za-z_\\{])(\d[\d,]*\.?\d*)', c)


def pointer(item, what, block, extra=()):
    ps = srcs(block) + [e for e in extra if e not in srcs(block)]
    if not ps:
        return ''
    return '%% pw1-draft-cuts (item %s): removed %s; its numbers stay in: %s\n' % (item, what, '; '.join(ps))


def once(t, s):
    n = t.count(s)
    assert n == 1, 'anchor occurs %d times (want 1): %r' % (n, s[:120])
    return t.index(s)


def op(item, what, old, new, drafts='both', ptr=True, extra=()):
    """Replace the literal `old` (exactly once) by `new`; `new` may use {PTR} for the pointer comment."""
    global text
    if drafts not in ('both', WHICH):
        return
    i = once(text, old)
    p = pointer(item, what, old, extra) if ptr else ''
    new = new.replace('{PTR}', p)
    rec = dict(item=item, what=what, kind='replace' if new.replace(p, '').strip() else 'remove',
               lines=[lineno(text, i), lineno(text, i + len(old) - 1)], words_removed=words(old),
               words_added=words(new), refs_removed=[r for r in refs(old) if r not in refs(new) or refs(old).count(r) > refs(new).count(r)],
               refs_added=[r for r in refs(new) if r not in refs(old)], labels_removed=[l for l in labels(old) if l not in labels(new)],
               numbers_left=sorted(set(nums(old)) - set(nums(new)), key=lambda x: nums(old).index(x)),
               pointer=p.strip(), before=old, after=new)
    report.append(rec)
    text = text[:i] + new + text[i + len(old):]


def span(start, end, include_end=False):
    """The literal text from `start` (unique) to the first `end` after it (end excluded unless include_end)."""
    i = once(text, start)
    j = text.index(end, i + len(start))
    return text[i:j + (len(end) if include_end else 0)]


# ---------------------------------------------------------------------------------------------
# Item 14. The abstract's maze-deletion / route-composition sentence (the Head's \TODO after it is untouched).
op(14, 'abstract sentence: deletion transfers on a colour maze; route composition and the plan-space verb',
   span("On a colour maze with the engine's integer judge, deletion transfers\n", '% w22 draft-rulings (Ruling 12)'),
   '{PTR}')

# Contribution 3: the route clause goes; the CMAZE-T sentences stay.
op(14, 'contribution 3: the maze deletion-transfer and route/verb clause',
   '\\item \\textbf{The same properties on a maze.} Deletion transfers exactly; composition on routes is\n'
   '  recovered by a plan-space verb, which we report as an \\emph{interface} result because it lifts a\n'
   '  matched-bytes lookup floor more than it lifts the n-CLU (Sec.~\\ref{sec:maze}). On a transition\n'
   '  instrument with symmetric inputs,',
   '\\item \\textbf{The same properties on a maze.} On a colour-maze transition instrument (CMAZE-T) with\n'
   '  symmetric inputs,', ptr=False)

# Thickness paragraph: the route sentence goes.
op(14, 'thickness paragraph: the route-task sentence',
   "On routes, where the engine's judge scores whole plans at three scored seeds, the result is\n"
   'a negative for the n-CLU and a positive for the interface. The transition instrument on the same\n'
   "engine, also under the engine's judge, prints six seeds",
   "The maze transition instrument, under the engine's judge, prints six seeds", ptr=False)

# The object section: the reason given for the frozen encoder rested on the CMAZE-2 autoencoder's bank overlap.
op(14, 'object section: the CMAZE-2 autoencoder evaluation-bank overlap (543 / 529 / 552)',
   '\\noindent The frozen encoder and decoder are declared outside the deletable object. This is\n'
   "forced, not chosen: the frozen autoencoder's own training set contains evaluation-bank items\n"
   '(543 / 529 / 552 on the three maze seeds), so the weights fitted with the deleted rows present\n'
   'are untouched by any deletion. % SRC: w13-seeds-and-tables/TABLES.md\n',
   '\\noindent The frozen encoder and decoder are declared outside the deletable object: they are\n'
   'frozen, so no deletion touches their weights, which may have been fitted with the deleted rows\n'
   'present.\n{PTR}', drafts='v2')
op(14, 'object section: the CMAZE-2 autoencoder evaluation-bank overlap (543 / 529 / 552)',
   '\\noindent We declare the frozen encoder and decoder outside the deletable object. The choice\n'
   "is forced: the frozen autoencoder's own training set contains evaluation-bank items\n"
   '(543 / 529 / 552 on the three maze seeds), so weights fitted with the deleted rows present are untouched by any deletion. % SRC: w13-seeds-and-tables/TABLES.md\n',
   '\\noindent We declare the frozen encoder and decoder outside the deletable object: being frozen,\n'
   'their weights, which may have been fitted with the deleted rows present, are untouched by any deletion.\n{PTR}',
   drafts='style')

# The maze section: CMAZE-1/2/2b definitions, Sec. maze-del + Tab. maze-del, Sec. req-layer, Sec. verb + Tab. verb,
# and the two-instrument lead-in of the world-model subsection. One section, CMAZE-T only.
MAZE_HEAD = '\\section{The same properties on a maze}\n\\label{sec:maze}\n' + BAR
op(14, 'maze section lead-in (CMAZE-1/2/2b/O names), Sec. maze-del with Tab. maze-del (CMAZE-2), Sec. req-layer '
       '(CMAZE-2b), Sec. verb with Tab. verb (routes, read verb), and the world-model subsection heading and its '
       'two-instrument lead-in',
   span(MAZE_HEAD, '\\paragraph{CMAZE-T, a minor contribution'),
   '\\section{The same properties on a maze}\n\\label{sec:maze}\\label{sec:maze-wm}\n' + BAR + '\n'
   "This section asks the n-CLU to \\emph{generate} a world model's transitions on a colour maze, under\n"
   "the engine's judge. Baselines are matched on read floats, the landscape controls and the blind columns\n"
   'are printed, and every row prints its seed count (Tabs.~\\ref{tab:cmazet} and~\\ref{tab:cmazet-sym},\n'
   'App.~\\ref{app:mazewm}).\n{PTR}\n')

# Item 17: the CMAZE-T deletion facts keep ONE sentence in the text (the one in "what is not claimed").
op(17, 'CMAZE-T symmetric-input paragraph: the second deletion clause (the deletion sentence of the '
       '"what is not claimed" paragraph stays, with the table rows)',
   'Every landscape\ncontrol stays below the n-CLU on every seed (highest 0.495, seed 5), and deletion stays exact on all\nsix seeds.\n',
   'Every landscape\ncontrol stays below the n-CLU on every seed (highest 0.495, seed 5).\n', ptr=False)

# CMAZE-O paragraph and the section's closing summary -> the one CMAZE-suite paragraph.
op(14, 'the CMAZE-O paragraph (withdrawn margin, control check) and the maze section\'s closing summary '
       '(CMAZE-2/2b deletion transfer, the verb, CMAZE-O)',
   span("against 71 to 140 s and 8,000 gradient steps for a baseline's retrain-without.\n% ENDSRC\n",
        BAR + '\\section{Limitations of the naive unit'),
   "against 71 to 140 s and 8,000 gradient steps for a baseline's retrain-without.\n% ENDSRC\n"
   'The run count, the configuration ladder and every negative with its failing link are in\nApp.~\\ref{app:mazewm}.\n\n'
   '{PTR}'
   '\\paragraph{Toward a CMAZE suite.} CMAZE-T is the first instrument of a planned CMAZE suite on the same\n'
   'engine and judge. The transition task will be extended to observation generation, to route planning\n'
   'and to the further tasks already prototyped, so that CMAZE as a whole becomes a useful benchmark for\n'
   'the world-models community.\n\n')

# Limitations: item 1 loses its route-task half; the CMAZE-2 post-deletion and maze-refusal items go.
op(14, 'limitation 1: "no n-CLU row on the route tasks" (CMAZE-2, CMAZE-2b)',
   '\\item \\textbf{No n-CLU row on the route tasks (CMAZE-2, CMAZE-2b).} On CMAZE-2 a matched-parameter transformer beats the\n'
   '  n-CLU on every composition row ($-0.0560$ / $-0.0209$ / $-0.0434$), and the landscape controls do\n'
   '  not collapse (landscape shuffled 0.9509 and random atoms 0.9601 against\n'
   '  1.0000 on the seen row), so no row is claimed for the n-CLU (App.~\\ref{app:cmaze2}). On\n'
   '  CMAZE-2b the controls again fail to collapse (Sec.~\\ref{sec:verb}). % SRC: w13-seeds-and-tables/TABLES.md\n'
   '  On CMAZE-T with symmetric inputs',
   "\\item \\textbf{On the maze transition task the n-CLU's lead is small and seed-dependent.}\n"
   '  {PTR}  On CMAZE-T with symmetric inputs')
op(14, 'limitations: post-deletion composition on the maze (CMAZE-2) and maze refusal as novelty detection',
   span('\\item \\textbf{Post-deletion composition on the maze fails', '\\item \\textbf{At most three seeds on Shapes3D.}'),
   '{PTR}')

# Leaving note: the route-task clause.
op(14, 'leaving note: composition on an engine-judged route task',
   'held-out combination over the strongest. It does not provide composition on an engine-judged route task, where the\n'
   'operator is task-shaped and belongs to the interface; on the transition task its lead over matched\n',
   'held-out combination over the strongest. On the maze transition task its lead over matched\n', ptr=False)

# App. K-curve: the comparison with the CMAZE-2 n-CLU's rank/width (0.938).
op(14, 'App. K-curve: the CMAZE-2 n-CLU rank/width comparison (0.938)',
   '0.4375--0.9844 on seed 2; at the maximum it exceeds the 0.938 at which a maze n-CLU was\n'
   'decorative, and the controls still collapse, so rank/width is a warning and only the control\n'
   'decides. % SRC: w12b-nr6-and-kstar/FINDINGS.md; w13-seeds-and-tables/TABLES.md\n',
   '0.4375--0.9844 on seed 2, and the controls still collapse, so rank/width is a warning and only the\n'
   'control decides. % SRC: w12b-nr6-and-kstar/FINDINGS.md; w13-seeds-and-tables/TABLES.md\n{PTR}', drafts='v2', extra=['w13-seeds-and-tables/TABLES.md'])
op(14, 'App. K-curve: the CMAZE-2 n-CLU rank/width comparison (0.938)',
   '0.4375--0.9844 on seed 2. At the maximum it exceeds the 0.938 at which a maze n-CLU was\n'
   'decorative, yet the controls still collapse; rank/width is therefore a warning, and only the\n'
   'control decides. % SRC: w12b-nr6-and-kstar/FINDINGS.md; w13-seeds-and-tables/TABLES.md\n',
   '0.4375--0.9844 on seed 2, yet the controls still collapse; rank/width is therefore a warning, and\n'
   'only the control decides. % SRC: w12b-nr6-and-kstar/FINDINGS.md; w13-seeds-and-tables/TABLES.md\n{PTR}', drafts='style', extra=['w13-seeds-and-tables/TABLES.md'])

# App. implementations: implementation 1 was the CMAZE-2 n-CLU; the two Shapes3D implementations renumber 1 and 2.
op(14, 'App. implementations: section heading (three -> two implementations)',
   '\\section{Exact deletion: three implementations, and the failure they agree on}',
   '\\section{Deletion on Shapes3D: two implementations, and the failure they agree on}', ptr=False)
op(14, 'App. implementations: Tab. impl-maze (implementation 1, the CMAZE-2 n-CLU)',
   span('\\begin{table}[h]\n\\centering\\small\n\\caption{\\textbf{Exact (n-CLU-scoped) deletion, implementation 1: the maze n-CLU (CMAZE-2).}',
        '\\end{table}\n\n', include_end=True),
   '{PTR}')
op(14, 'App. implementations: Tab. impl caption (renumbered; the comparison with the maze band)',
   'deletion on Shapes3D, implementations 2 and 3, bands not\nshared.} Seeds and configurations are separate fields; the nine legs of row 3 are 3 configurations\n'
   "$\\times$ 3 seeds. Implementation 2's band is five orders coarser than the maze n-CLU's\n(Tab.~\\ref{tab:impl-maze}).}",
   'deletion on Shapes3D, implementations 1 and 2, bands not\nshared.} Seeds and configurations are separate fields; the nine legs of row 2 are 3 configurations\n'
   '$\\times$ 3 seeds.}', ptr=False)
op(14, 'App. implementations: Tab. impl rows renumbered',
   '2. Shapes3D downdate build (TEST), assertion battery & 3 & 1 & 0 &',
   '1. Shapes3D downdate build (TEST), assertion battery & 3 & 1 & 0 &', ptr=False)
op(14, 'App. implementations: Tab. impl rows renumbered',
   '3. Shapes3D single-design candidates (TEST), assertion battery &',
   '2. Shapes3D single-design candidates (TEST), assertion battery &', ptr=False)
op(14, 'App. implementations: paragraph heading renumbered',
   '\\paragraph{Implementation 2 in full.}', '\\paragraph{Implementation 1 in full.}', ptr=False)

# The three appendices of item 14 (CMAZE-2 in full, CMAZE-1 at the harness tier, the read verb in full) and the
# CMAZE-2 refusal-display appendix. The CMAZE-T conditions figure (item 1's) sat inside the read-verb appendix: it
# is lifted out first and re-seated at the head of the CMAZE-T appendix; its item-15 \ref is taken out on the way.
op(15, 'Fig. maze-wm caption: the \\ref to Fig. maze-rel-tuples (the tie keeps its table pointer)',
   '(0.9004, on every held-out tuple, Fig.~\\ref{fig:maze-rel-tuples}); on seed 5 the best matched',
   '(0.9004, on every held-out tuple, Tab.~\\ref{tab:cmazet-sym}); on seed 5 the best matched', ptr=False)
COND = span('% SRCBLOCK: w16-maze-wm/t1/BARS-A1-test.json', '}{fig:maze-wm}\n% ENDSRC\n', include_end=True)
op(14, 'App. read verb: the CMAZE-T conditions figure (fig:maze-wm) lifted out of the removed appendix, re-seated in App. mazewm',
   COND + '\n', '', ptr=False)
report[-1]['kind'] = 'move'
op(14, 'App. CMAZE-2 in full (Tab. cmaze2), App. CMAZE-1 proper (Tab. cmaze1), App. the read verb in full '
       '(Tabs. req, req2b, maze-del-2b; Fig. verb = fig_maze_readverb; the fig_maze_requestlayer note) with '
       'Fig. maze-rel-tuples (item 15) inside it, and App. maze refusal displays (CMAZE-2)',
   span(SEP + '\\section{CMAZE-2 in full', SEP + '\\section{The zero-dynamics identity'),
   '{PTR}')
op(14, 'App. mazewm: the conditions figure re-seated at the head of the CMAZE-T appendix',
   '\\label{app:mazewm}\n\n', '\\label{app:mazewm}\n\n' + COND + '\n', ptr=False)
report[-1]['kind'] = 'move'

# App. maze world-model rows: CMAZE-O goes everywhere in it.
op(14, 'App. mazewm: "both harnesses" (CMAZE-O harness)',
   '\\paragraph{The harness smokes.} Both harnesses shipped ready.',
   '\\paragraph{The harness smoke.} The harness shipped ready.', ptr=False)
op(14, 'App. mazewm: the CMAZE-O harness smoke',
   span('% SRCBLOCK: w15-cmaze-o-harness/FINDINGS.md\n', '% SRCBLOCK: w16-maze-wm/FINDINGS.md; w15-cmaze-t-harness/FINDINGS.md\n'),
   '{PTR}')
op(14, 'App. mazewm: control defect, CMAZE-O half',
   '\\emph{A control defect these smokes carry.} Both harnesses shipped\n',
   '\\emph{A control defect this smoke carries.} The harness shipped\n', ptr=False)
op(14, 'App. mazewm: control defect, the \\ref to Tab. cmazeo',
   'every across-wells cell in\nTabs.~\\ref{tab:cmazet} and~\\ref{tab:cmazeo} uses the derangement.',
   'every across-wells cell in\nTab.~\\ref{tab:cmazet} uses the derangement.', ptr=False)
op(14, 'Tab. cmazet caption: the "contains no CMAZE-O, CMAZE-1, CMAZE-2 or CMAZE-2b row" sentence',
   'Values are mean (seed 0 / 1 / 2). This table contains\nno CMAZE-O, CMAZE-1, CMAZE-2 or CMAZE-2b row.}',
   'Values are mean (seed 0 / 1 / 2).}', ptr=False)
op(17, 'panel-figure comment: three -> two placeholders',
   "% The three PLACEHOLDERs name the per-arm outputs of Table tab:cmazet's own arms, which w16 never persisted.",
   "% The two PLACEHOLDERs name the per-arm outputs of Table tab:cmazet's own arms, which w16 never persisted\n"
   '% (a third, in the post-deletion panel figure, left the draft with that figure: pw1-draft-cuts, item 17).', ptr=False)
op(17, 'Fig. cmazet-deletion (fig_cmazet_deletion): slot, caption and its PLACEHOLDER',
   span('% SOURCE ARTEFACT: figures/fig_cmazet_deletion.pdf\n', '}{fig:cmazet-deletion}\n', include_end=True),
   '% pw1-draft-cuts (item 17): removed Fig. cmazet-deletion (figures/fig_cmazet_deletion.pdf; the image file is\n'
   '% left in place); its facts stay in Tabs. cmazet and cmazet-sym (deletion rows) and one sentence of Sec. maze;\n'
   '% records: figures/data/cmazet_panels.json; figures/data/cmazet_items.REPRO.json; w19-maze-t-symmetric/test/BOTH_test_seed0.json\n')
op(14, 'Tab. cmazet-sym caption: "no CMAZE-O, CMAZE-1, CMAZE-2 or CMAZE-2b row"',
   'Tab.~\\ref{tab:cmazet} and no CMAZE-O, CMAZE-1, CMAZE-2 or CMAZE-2b row.}',
   'Tab.~\\ref{tab:cmazet}.}', ptr=False)
op(14, 'Tab. cmazeo (CMAZE-O, observation generation)',
   span('\\begin{table}[h]\n\\centering\\scriptsize\n\\caption{\\textbf{Maze world-model rows: CMAZE-O', '\\end{table}\n', include_end=True),
   '{PTR}')
op(14, 'App. mazewm, configuration ladder and run count: the CMAZE-O ladder and the eighteen-run count',
   span('\\emph{CMAZE-O}: the autoencoder-latent design (rank/width at most 0.3) failed validation, its\n',
        '(validation only, not winners) and any seed beyond three are also not run.\n', include_end=True),
   'Deletion rows for the match-bit design (validation only, not a winner) and any seed beyond three are\nnot run.\n{PTR}',
   extra=['w16-maze-wm/FINDINGS.md', 'w16-maze-wm/TABLES.md'])
op(14, 'App. mazewm: the fallback route task paragraph (CMAZE-2b; refs to Tabs. verb and maze-del-2b)',
   span('\\paragraph{The fallback route task.}', '\\paragraph{Declared negatives, each with its failing link.}'),
   '{PTR}')
op(14, 'App. mazewm, declared negatives: the CMAZE-O half',
   'the match-bit\ndesign (\\emph{interface}). CMAZE-O: the autoencoder-latent design (\\emph{content/read}: the autoencoder latent is not carried by\n'
   'a low-rank n-CLU); the field design against converged baselines (\\emph{interface}); the second latent design (\\emph{read}: the\n'
   "latent pipeline caps the n-CLU). CMAZE-O therefore has no single failing link, and its field design's\n"
   'control check holds while its composition-over-baselines claim is withdrawn.\n',
   'the match-bit\ndesign (\\emph{interface}).\n', ptr=False)

# Item 17: every other \ref to Fig. cmazet-deletion.
op(17, 'Fig. task-cmazet caption: figure range ending at Fig. cmazet-deletion',
   '(Figs.~\\ref{fig:cmazet-families}--\\ref{fig:cmazet-deletion})',
   '(Figs.~\\ref{fig:cmazet-families}--\\ref{fig:cmazet-door-tint})', ptr=False)
op(17, 'App. placeholder index: three rows -> two (Fig. cmazet-deletion)',
   'Three rows remain, each marked in the text by a red \\textsc{placeholder} whose argument names\n'
   'the row: the per-arm generated states of the arms of Tab.~\\ref{tab:cmazet} in\n'
   'Figs.~\\ref{fig:cmazet-families}, \\ref{fig:cmazet-door-tint} and~\\ref{fig:cmazet-deletion}. They are',
   'Two rows remain, each marked in the text by a red \\textsc{placeholder} whose argument names\n'
   'the row: the per-arm generated states of the arms of Tab.~\\ref{tab:cmazet} in\n'
   'Figs.~\\ref{fig:cmazet-families} and~\\ref{fig:cmazet-door-tint}. They are', ptr=False)
op(17, 'App. placeholder index: the panel-figure list (Fig. cmazet-deletion)',
   'Figs.~\\ref{fig:cmazet-families}, \\ref{fig:cmazet-per-arm}, \\ref{fig:cmazet-door-tint}\n'
   'and~\\ref{fig:cmazet-deletion} draw every arm of the words-plus-observation condition from its\n'
   'persisted objects; their three placeholders',
   'Figs.~\\ref{fig:cmazet-families}, \\ref{fig:cmazet-per-arm}\n'
   'and~\\ref{fig:cmazet-door-tint} draw every arm of the words-plus-observation condition from its\n'
   'persisted objects; their two placeholders', ptr=False)

# App. declared not-runs: the CMAZE-2 post-deletion repair and the plan-versus-latent (verb) figure.
op(14, 'App. not-runs: the repaired maze post-deletion composition (CMAZE-2)',
   '\\item The repaired maze post-deletion composition on TEST at 3 seeds (one validation measurement\n  exists).\n', '{PTR}',
   extra=['w13-seeds-and-tables/TABLES.md'])
op(14, 'App. not-runs: the plan-versus-latent (read verb) figure',
   '\\item A figure for the plan-versus-latent result, for the retention law, and for the $K=5\\to\\Kstar$\n',
   '\\item A figure for the retention law and for the $K=5\\to\\Kstar$\n', ptr=False)

open(OUT, 'w').write(text)
json.dump(dict(input=IN, output=OUT, draft=WHICH, ops=report), open(REP, 'w'), indent=1)
print('%s: %d ops, %d -> %d lines' % (WHICH, len(report), orig.count('\n'), text.count('\n')))
