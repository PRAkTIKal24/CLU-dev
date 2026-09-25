"""pw2_draft_numbers/numbers_left.py, adapted by pw2-draft-leftovers (part 7): the limitations list is no longer Head-held
in either draft (B3 borrowing), so its "lock-step" class is removed and it is classified like any other text; three
identifier rules added for what the list keeps ($K\\ge36$, $K=24$, rank 64).
NUMBERS-LEFT: every number remaining in the main text of each draft, with a justification class, and counts per section.
usage: python3 numbers_left.py V2.tex STYLE.tex > NUMBERS-LEFT.md ; exit 1 if any row is UNCLASSIFIED."""
import re, sys
from collections import Counter, OrderedDict
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', 'pw2_draft_numbers'))
from scan import scan

ID = [  # (regex on the window around the token, justification) -- identifiers and protocol constants, not results
 (r'131,072 float32', 'definition: parameters at the byte reference (524,288 / 4), part of the same definition'),
 (r'524,288-byte figure', 'definition: the byte reference, defined once here (tables cite it)'),
 (r'[Ss]eeds? [0-5](, [0-5])*( and [0-5])?\b', 'identifier: seed index'),
 (r'Kstar=32', 'identifier: the design name $K^\\star=32$'),
 (r'\$K=5\$|K=5\b', 'identifier: the $K=5$ endpoint (a design name)'),
 (r'depth at 6|\$n=[\d,]+|n=3,4,5,6|n\\ge3|n=3\\to6|rungs? 3--6|\d-way|\$n=\d', 'identifier: ladder rung / hold-out order ($n$)'),
 (r'rows 3--4|rows? [34]\b', 'identifier: hold-out row name (row 3 = pairs, row 4 = six-factor)'),
 (r'95\\%', 'identifier: interval level (95% bootstrap interval)'),
 (r'10\\times', 'identifier: the schedule name "$10\\times$ training steps"'),
 (r'2024 journal', 'identifier: year in a citation phrase'),
 (r'5\\times5', 'identifier: panel crop size (drawing rule)'),
 (r'pocket 2', 'identifier: outcome code (legal, move south, pocket 2)'),
 (r'\d+-atom', 'identifier: design name (6-atom base design, 10-atom relational design)'),
 (r'random\s*1\\%|^1\\% of items', 'identifier: the item-deletion scope name ("random 1%")'),
 (r'h/h\^\\star\\in\[0\.5,2\]', 'identifier: named bucket $h/h^\\star\\in[0.5,2]$'),
 (r'\\lambda=0|\^\{-1\}', 'identifier: mathematical notation'),
 (r's_\{t\+1\}', 'identifier: mathematical notation ($s_{t+1}$)'),
 (r'float(64|32)', 'identifier: tokeniser artefact inside the type name float64 / float32'),
 (r'\(([1-4])\) the', 'identifier: list enumerator (1)-(4)'),
 (r'3-seed', 'identifier: "3-seed mean" (protocol)'),
 (r'\(3 seeds', 'protocol: seed count'),
 (r'1\.0 tuple', 'protocol: the declared margin, defined once in Methods'),
 (r'64 checks', 'protocol: size of the read-path assertion (no table carries it)'),
 (r'256 requests', 'protocol: size of the bit-identity assertion (no table carries it)'),
 (r'share is 1 by construction', 'definition: coverage share 1 by construction (not a measurement)'),
 (r'about 0\.6', 'rounded high-level (ROUNDED.md: 0.610083 / 0.614277)'),
 (r'K\\ge36', 'identifier: the partition range $K\\ge36$ (the partitions that fit the byte reference; the measured one is in Tab. kcurve caption)'),
 (r'\$K=24\$', 'identifier: partition name $K=24$ (location of the footprint minimum; its value is in Tab. kcurve)'),
 (r'^\s*64 on any swept', 'identifier: rank 64 (design constant of the swept objects)'),
 (r'just under 1\.4', 'rounded high-level (ROUNDED.md: 1.358 to 1.372)'),
]
ROUNDED_WORDS = [
 (r'about (seven|six|thirty|forty|one|four)|nearly (five|a gigabyte|all|two thirds)|about four thousand|nine in ten|half a percent|up to about four|two thirds|more than eighteen|far below one|less than one|about a (tenth|third)|by about one', 'rounded high-level (ROUNDED.md)'),
]
PRONOUN = [(r'established one|the one our|never-inserted one|the one measured', 'not a number: pronoun / determiner "one"')]
def classify(r):
    if r['headheld']: return 'Head-held (%s)' % r['headheld']
    c = r['context']; a, b = max(0, r['pos'] - 30), r['pos'] + len(r['token']) + 30
    w = c[a:b]
    lo, hi = r['pos'], r['pos'] + len(r['token'])
    for rx, j in PRONOUN + ID + ROUNDED_WORDS:   # the rule applies only if its match covers the token itself
        for m in re.finditer(rx, c):
            if m.start() <= lo and hi <= m.end(): return j
    if r['kind'] == 'word':
        if re.search(r'zero-(dynamics|step)', w): return 'identifier: the named zero-dynamics limit / zero-step arm'
        if re.search(r'three properties|three things|The three pull', w): return 'count: the three properties (the paper\'s frame)'
        if re.search(r'(three|six) (scored )?seeds|of (three|six) seeds|all (three|six)|two of three|three seeds|six seeds|on two$|on two of|level with it on two|behind on one|every seed but one', w):
            return 'protocol / high-level seed count'
        return 'count word, high-level (not table-readable)'
    return 'UNCLASSIFIED'

def sentence(r):
    c = r['context'].strip()
    return c if len(c) <= 140 else c[:137] + '...'

out = []; P = out.append; bad = 0
P('# NUMBERS-LEFT — every number remaining in the main text (title through the end of Discussion), both drafts')
P('')
P('Generated by `pw2_draft_numbers/numbers_left.py` from the installed drafts. Main text = the lines from `\\title` to the '
  '`AI use statement` heading; the abstract is included; tables, figures, `\\figslot` arguments (captions) and LaTeX comments are excluded; '
  '`\\ref`/`\\label`/`\\cite` keys are masked. "digit" = every digit run (the provenance checker\'s tokeniser, `\\sci{m}{e}` whole); '
  '"word" = a spelled-out cardinal (zero … twelve, twenty, thirty, forty, hundred, thousand, half, twice, …). Nothing is filtered: '
  'identifiers, seed indices and pronoun "one" are listed and classed, not dropped.')
P('')
for path in sys.argv[1:3]:
    rows = scan(path); name = path.split('/')[-1]
    for r in rows: r['just'] = classify(r)
    bad += sum(r['just'] == 'UNCLASSIFIED' for r in rows)
    P('## %s' % name); P('')
    cnt = OrderedDict()
    for r in rows:
        k = r['section']; cnt.setdefault(k, Counter())
        cls = 'Head-held' if r['just'].startswith(('Head-held', 'lock-step')) else r['just'].split(':')[0].split(' (')[0]
        cnt[k][(r['kind'], cls)] += 1
    P('### Count per section (digit tokens / word tokens, by class)'); P('')
    P('| section | digit | word | Head-held | identifier / protocol / definition | rounded high-level | count word / seed count | not a number | UNCLASSIFIED |')
    P('|---|---|---|---|---|---|---|---|---|')
    for k, c in cnt.items():
        tot = lambda f: sum(v for (kind, cl), v in c.items() if f(kind, cl))
        P('| %s | %d | %d | %d | %d | %d | %d | %d | %d |' % (k, tot(lambda a, b: a == 'digit'), tot(lambda a, b: a == 'word'),
          tot(lambda a, b: b == 'Head-held'), tot(lambda a, b: b in ('identifier', 'protocol', 'definition')),
          tot(lambda a, b: b.startswith('rounded')), tot(lambda a, b: b.startswith('count') or b.startswith('protocol / high')),
          tot(lambda a, b: b.startswith('not a number')), tot(lambda a, b: b == 'UNCLASSIFIED')))
    P('')
    ab = [r for r in rows if r['section'] == 'Abstract']
    P('### Abstract: every numeral (limit: at most five numbers)'); P('')
    for r in ab: P('- line %d, %s `%s` — %s' % (r['line'], r['kind'], r['token'], r['just']))
    q = [r for r in ab if not (r['just'].startswith('identifier') or r['just'].startswith('not a number'))]
    P(''); P('**Quantities in the abstract: %d** (%s), Head-held among them: %d. Identifier uses (not counted): %d; pronoun "one" (not a number): %d.' % (
        len(q), ', '.join('"%s"' % r['token'] for r in q), sum(1 for r in q if r['headheld']),
        sum(1 for r in ab if r['just'].startswith('identifier')), sum(1 for r in ab if r['just'].startswith('not a number'))))
    P('')
    P('### Every number'); P('')
    P('| line | section | kind | number | justification | line text (code) |'); P('|---|---|---|---|---|---|')
    for r in rows:
        P('| %d | %s | %s | `%s` | %s | `%s` |' % (r['line'], r['section'], r['kind'], r['token'], r['just'], sentence(r).replace('|', '\\|').replace('`', "'")))
    P('')
print('\n'.join(out))
print('\nUNCLASSIFIED rows: %d' % bad)
sys.exit(1 if bad else 0)
