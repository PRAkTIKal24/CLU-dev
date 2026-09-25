#!/usr/bin/env python3
"""PW-1 figure install, text side: items 1 (1.5, 8), 9, 10, 11, 12 (12.7), 16 (16.1) in one draft.

usage: edit.py IN.tex OUT.tex REPORT.json

Every operation is anchored on literal text (whitespace-tolerant) and asserts that its anchor occurs
exactly once, so a drifted draft fails loudly. The operations are the same for DRAFT-v2 and its style
twin: every anchored region is byte-identical in the two drafts. New captions and paragraphs are the
figure spokes' CAPTIONS.tex / TEXT-INSERTS.tex / DELETION-FACTS.tex wording, kept in blocks/*.tex with
SRC comments that name real files. Numbers that leave the paper keep a pointer in a
`% pw1-draft-figures (item N): removed ...; its numbers stay in: ...` comment (the provenance keyword is
avoided there, as in pw1_draft_cuts, so the style diff-check compares only surviving provenance).
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
IN, OUT, REP = sys.argv[1:4]
text = open(IN).read()
report = []


def block(name):
    return open(os.path.join(HERE, 'blocks', name)).read()


def pat(lit):
    parts = re.split(r'(\s+)', lit)
    return ''.join(r'\s+' if p.strip() == '' and p else re.escape(p) for p in parts)


def code_only(s):
    return '\n'.join(re.split(r'(?<!\\)%', l, 1)[0] for l in s.split('\n'))


def nums(s):
    c = re.sub(r'\\(ref|label)\{[^}]*\}', ' ', code_only(s))
    c = re.sub(r'\\sci\{([^}]*)\}\{([^}]*)\}', r' \1e\2 ', c)
    return re.findall(r'(?<![A-Za-z_\\{])(\d[\d,]*\.?\d*(?:e-?\d+)?)', c)


def op(item, what, old, new, regex=False, ptr=None):
    """Replace `old` (exactly once) by `new`. ptr: list of records the leaving numbers stay in."""
    global text
    p = old if regex else pat(old)
    ms = list(re.finditer(p, text, re.S))
    assert len(ms) == 1, '%s: anchor occurs %d times (want 1): %r' % (what, len(ms), old[:100])
    m = ms[0]
    was = m.group(0)
    left = [n for n in nums(was) if n.rstrip('.,') not in [x.rstrip('.,') for x in nums(new)]]
    assert ('{PTR}' in new) == bool(ptr), what
    new = new.replace('{PTR}', '%% pw1-draft-figures (item %s): removed %s; its numbers stay in: %s\n'
                      % (item, what, '; '.join(ptr or [])))
    line = text.count('\n', 0, m.start()) + 1
    text = text[:m.start()] + new + text[m.end():]
    report.append(dict(item=item, what=what, line_before=line, before=was, after=new,
                       numbers_leaving=left))


def figslot(stem, label):
    """regex for a whole \\figslot{stem}...}{label} call, through the end of its last line."""
    return r'\\figslot\{%s\}.*?\}\{%s\}[^\n]*' % (re.escape(stem), re.escape(label))


CEIL_PTR = ['w21-maze-t-seed4/FINDINGS.md', 'w21-maze-t-seed4/TABLES.md', 'w21-words-full-test/FINDINGS.md',
            'w21-rel-newseeds/FINDINGS.md']

# ---------------------------------------------------------------- item 1.5 / 8: the per-seed ceiling leaves
op('1.5', 'contribution bullet: "and a measured per-seed ceiling"',
   'six seeds and a measured per-seed ceiling, the n-CLU is ahead of',
   'six seeds, the n-CLU is ahead of')
op('1.5', 'contribution bullet: "at the ceiling"',
   'level with it at the ceiling on two and behind on one (mean',
   'level with it on two and behind on one (mean')
op('1.5', 'instrument paragraph: "every row prints its measured per-seed ceiling"',
   'terms or the colour identity are printed as falsifiers; and every row prints its measured per-seed\nceiling. The task is small,',
   'terms or the colour identity are printed as falsifiers. The task is small,')
op('1.5', 'paragraph title "with the ceiling beside every seed"',
   r'\paragraph{CMAZE-T with symmetric inputs: six seeds, with the ceiling beside every seed.}',
   r'\paragraph{CMAZE-T with symmetric inputs: six seeds.}')
op('1.5', 'results paragraph: "two ties at the ceiling"',
   'clear wins (seeds 0, 2 and 3), two ties at the ceiling (seeds 1 and 4,',
   'clear wins (seeds 0, 2 and 3), two ties (seeds 1 and 4,')
op('1.5 / 8', 'the per-seed ceiling paragraph (ceiling values, unsupported-row counts, cap), replaced by the hold-out mechanism',
   r'% SRCBLOCK: w21-maze-t-seed4/FINDINGS.md; w21-maze-t-seed4/TABLES.md; w21-rel-newseeds/FINDINGS.md\n'
   r'\\paragraph\{The measured per-seed ceiling, a ceiling for identity-coded readers\.\}.*?'
   r'recommend per-seed reporting of the ceiling to anyone who uses the task\.\n% ENDSRC\n',
   '{PTR}' + block('holdout_mechanism.tex'), regex=True, ptr=CEIL_PTR)
op('1.5', 'limitation item: "at the ceiling"',
   'level at the ceiling on two and behind on one, and behind on every seed of the full word design',
   'level on two and behind on one, and behind on every seed of the full word design')
# Tab. cmazet-sym
op('1.5', 'Tab. cmazet-sym caption: the definition of the ceiling',
   "step counts (8,000 and 80,000 steps) on that seed. ``Ceiling'' is the best score of any arm % SRC: w21-maze-t-seed4/FINDINGS.md\n"
   "fitted to the seed's training data, the unmatched ten-times-read-floats diagnostic included and the\n"
   "rule-given planner columns excluded. \\texttt{+rel}:",
   "step counts (8,000 and 80,000 steps) on that seed. % SRC: w21-maze-t-seed4/FINDINGS.md\n\\texttt{+rel}:")
op('1.5', 'Tab. cmazet-sym, words full design: the ceiling cells',
   '; per-seed best baseline 0.827 / 0.567 / 0.834; ceiling 0.845 / 0.567 / 0.866 \\\\',
   '; per-seed best baseline 0.827 / 0.567 / 0.834 \\\\')
op('1.5', 'Tab. cmazet-sym, observation: the ceiling cells',
   'behind on every seed; ceiling 0.479 / 0.467 / 0.477 / 0.458 / 0.417 / 0.485; the extractor',
   'behind on every seed; the extractor')
op('1.5', 'Tab. cmazet-sym, words + observation: "two ties at the ceiling"',
   'three clear wins, two ties at the ceiling, one loss.',
   'three clear wins, two ties, one loss.')
op('1.5', 'Tab. cmazet-sym: the per-seed ceiling row (whole), and the ceiling cells of the full-word-design row (0.845 / 0.567 / 0.866) and the observation row (0.479 / 0.467 / 0.477 / 0.458 / 0.417 / 0.485)',
   r'\\quad per-seed ceiling & 6 & [^\n]*\\\\\n',
   '{PTR}', regex=True, ptr=CEIL_PTR)
op('1.5', 'Tab. cmazet-sym, n-CLU-REL: "reaches the ceiling on every seed"',
   '; the n-CLU reaches the ceiling on every seed. An input-representation result',
   '. An input-representation result')
op('1.5', 'Tab. cmazet-sym, baseline training steps: the diagnostic that entered only the ceiling',
   'both printed and the better one used; a ten-times-read-floats diagnostic enters the ceiling only \\\\',
   'both printed and the better one used \\\\')

# ---------------------------------------------------------------- items 1, 8, 16: maze captions and the panel paragraph
MAZE_PTR = ['w21-maze-t-seed4/FINDINGS.md', 'w21-rel-newseeds/FINDINGS.md', 'w21-maze-t-seed4/TABLES.md']
op('1', 'the fig:maze-wm caption (rule text, ceiling, hard-seed split, per-seed relational values), rewritten',
   figslot('fig_maze_conditions', 'fig:maze-wm'), '{PTR}' + block('maze_wm.tex').rstrip('\n'), regex=True,
   ptr=MAZE_PTR)
PANEL_PTR = ['../../outputs/w17-paper/figures/data/CMAZET-AMBIGUITY.json', '../../outputs/w17-paper/figures/data/cmazet_items_rel.REPRO.json']
op('16', 'the fig:cmazet-families caption, rewritten', figslot('fig_cmazet_families', 'fig:cmazet-families'),
   '{PTR}' + block('cmazet_families.tex').rstrip('\n'), regex=True, ptr=PANEL_PTR)
op('16', 'the fig:cmazet-per-arm caption, rewritten', figslot('fig_cmazet_per_arm', 'fig:cmazet-per-arm'),
   block('cmazet_per_arm.tex').rstrip('\n'), regex=True)
op('16', 'the fig:cmazet-rel-per-arm caption, rewritten', figslot('fig_cmazet_rel_per_arm', 'fig:cmazet-rel-per-arm'),
   block('cmazet_rel_per_arm.tex').rstrip('\n'), regex=True)
op('16', 'the fig:cmazet-door-tint caption, rewritten', figslot('fig_cmazet_door_then_tint', 'fig:cmazet-door-tint'),
   block('cmazet_door_tint.tex').rstrip('\n'), regex=True)
op('16.1', 'insert: the paragraph that reads the panel figures (once, in Sec. maze-wm)',
   'The run count, the configuration ladder and every negative with its failing link are in\nApp.~\\ref{app:mazewm}.\n',
   'The run count, the configuration ladder and every negative with its failing link are in\nApp.~\\ref{app:mazewm}.\n\n'
   + block('reading_panels.tex'))

# ---------------------------------------------------------------- items 10, 11, 12, 2-3: Shapes captions
op('10', 'the fig:ladder caption (three panels), rewritten for the single panel', figslot('fig_ladder', 'fig:ladder'),
   '{PTR}' + block('ladder.tex').rstrip('\n'), regex=True,
   ptr=['_thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.rival.json', '_thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.json',
        'w12-composition-ladder/runs/shipped_seed0/rows_learned_ref_ownpool.json', '_thinkpad/w4/spoke-a-rivals/REPORT.md', 'w21-ladder-rivals-32k/CAPTION.md', 'Tab. ladder'])
op('11', 'the fig:crossover caption, rewritten', figslot('fig_crossover', 'fig:crossover'),
   block('crossover.tex').rstrip('\n'), regex=True)
op('12', 'the fig:deletion caption (six panels), rewritten for two', figslot('fig_deletion', 'fig:deletion'),
   '{PTR}' + block('deletion.tex').rstrip('\n'), regex=True, ptr=['w20-shapes-data-matched/SHAPES-MATCHED.json', 'Tab. del'])
op('2-3', 'the fig:adapt caption, rewritten to the image', figslot('fig_encoder_adapt', 'fig:adapt'),
   block('adapt.tex').rstrip('\n'), regex=True)

# ---------------------------------------------------------------- item 12.7: deletion facts into Tab. del + one sentence
op('12.7', 'Tab. del (c), value row: request counts added',
   'value bit cleared (1 bit, 0 rows read): red requests refused 1.000, false refusal on the other requests 0.000, every seed}',
   'value bit cleared (1 bit, 0 rows read): red requests refused 1.000 (720 requests per seed), false refusal on the other requests 0.000 (4,968 requests), every seed}')
op('12.7', 'Tab. del (c), red sphere row: request count added',
   'refusal \\emph{undefined}, no rows-3--4 combination requests the red sphere; false refusal 0.000}',
   'refusal \\emph{undefined}, no rows-3--4 held-out combination requests the red sphere; false refusal 0.000 (5,688 requests), every seed}')
op('12.7 / 13', 'the text sentence on measurements (a), (b), (d), in item-13 phrasing with the measured bounds',
   "\\noindent Measurements (a) and (b) are exact on every scope and seed, and (d)'s check is exact too:\n"
   "the downdated object and the object refit without the rows compose identically on every combination of\n"
   "all nine legs.",
   "% SRCBLOCK: w20-shapes-data-matched/SHAPES-MATCHED.md\n"
   "\\noindent After deletion the n-CLU matches an n-CLU that was never fitted on the deleted rows, and its address\n"
   "book matches a book refit without them: over every scope and seed the content's largest mode\n"
   "deviation is at most \\sci{5.38}{-07}, the book's statistic deviates by 0 and its rendered features by\n"
   "at most \\sci{7.81}{-08} (relative); the frozen encoder and decoder are outside the object. Check (d)\n"
   "holds as well: the downdated object and the object refit without the rows compose identically on every\n"
   "combination of all nine legs.\n% ENDSRC\n")

# ---------------------------------------------------------------- item 9: the rendered Shapes3D figures
op('9', 'the w22 Ruling-2 comment block on the removed T7 rendered slots',
   r'% ---- w22 draft-rulings \(Ruling 2\): the three T7 rendered success/failure panel slots are removed\..*?'
   r'is a figures-owner item \(declared in app:notrun\)\.\n',
   '% ---- w22 draft-rulings (Ruling 2) removed the three T7 rendered slots (rc_w19_t7_*, retired 16,384-row comparators);\n'
   '% pw1-draft-figures (item 9) re-renders them against the data-matched baselines of this table: see after the next paragraph.\n',
   regex=True)
op('9', 'insert: the rendered figures and their paragraph (Sec. shapes-t7)',
   '\n\n\\paragraph{A second, earlier baseline set.}',
   '\n\n' + block('shapes_rendered.tex') + '\n\\paragraph{A second, earlier baseline set.}')
op('9', 'declared not-run: the re-render (now run)',
   '\\item A re-render of the Shapes3D per-arm success and failure panels against the data-matched baselines.\n',
   '% pw1-draft-figures (item 9): the re-render against the data-matched baselines was run (Figs. shapes-per-model,\n'
   '% shapes-baseline-wins); its not-run item is removed.\n')

open(OUT, 'w').write(text)
json.dump(report, open(REP, 'w'), indent=1)
print('%d operations applied' % len(report))
