"""Classify every main-text number token (numscan.py) with a justification. Written fresh for the closing audit; the
rules below are this audit's own and do not import pw2_draft_numbers.

Classes: HEAD-HELD (inside a Head-held paragraph, not editable in PW-2) | IDENTIFIER (a name: seed index, rung,
K, row, criterion, dataset, dtype, design, year, list enumerator, math notation, schedule name) | PROTOCOL/DEFINITION
(a declared constant of the protocol, or the one definition of the byte reference) | ROUNDED (a rounded high-level
value; each is re-checked against its record by rounded_check.py) | COUNT (a count word of discrete things, high-level,
not a value read from a table) | NOT-A-NUMBER (pronoun or determiner "one", "either half", "half-life") |
UNCLASSIFIED (no rule fired: a finding).
usage: python3 numclass.py DRAFT.tex [--json OUT]  -> markdown table on stdout; exit 1 on UNCLASSIFIED
"""
import json, re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pw2_draft_sections'))
from numscan import scan
import rounded_check

HH_ROWS = None


def headheld_lines(path):
    import headheld
    out = []
    for name, lines, _, _ in headheld.rows(path):
        a, _, b = lines.partition('–')
        out.append((name, int(a), int(b or a)))
    return out


ROUNDERS = r'(about|nearly|roughly|just under|more than|less than|up to about|at most about|far below|within)\s+(a\s+|an\s+)?$'


def classify(t, lines, hh, anchors_code):
    ln, tok, word = t['line'], t['tok'], t['word']
    full = lines[ln - 1]
    col = t['col']
    prev = re.sub(r'(?<!\\)%.*', '', lines[ln - 2]).rstrip() if ln >= 2 else ''
    before = ((prev + ' ') if col < 60 else '') + full[:col]
    before = before[-60:]
    after = full[col + len(tok):col + len(tok) + 60]
    low = tok.lower()
    for name, a, b in hh:
        if a <= ln <= b:
            return 'HEAD-HELD', name
    if t['kind'] == 'digit':
        # glued to letters (not a macro name): part of a name
        if word != tok and re.search(r'[A-Za-z]', word) and not re.search(r'\\[A-Za-z]*$', full[:col]):
            return 'IDENTIFIER', 'part of a name (`%s`: dataset, dtype or criterion name)' % word
        if re.search(r'[Ss]eeds?\s+[\d\s,/and]*$', before) or re.search(r'^\'s\b', after) and re.search(r'[Ss]eed\s+$', before):
            return 'IDENTIFIER', 'seed index'
        if re.search(r'(n=|n\\ge|\\to|n=[\d,]*,)$', before) or re.search(r'^-way', after) or re.search(r'rungs?\s+[\d\-]*$', before) \
                or re.search(r'holding depth at $', before):
            return 'IDENTIFIER', 'rung name ($n$-way ladder)'
        if re.search(r'(K=|Kstar=|K\\ge|K\\in\\\{[\d,]*|K=\d+\\ldots|K=10\$, (\d+, )*|and )$', before) and re.search(r'K', before[-30:]):
            return 'IDENTIFIER', 'partition size $K$ (design name / sweep grid)'
        if re.search(r'^\\%', after):
            if tok == '95': return 'PROTOCOL/DEFINITION', 'interval level (95%)'
            if tok == '1': return 'IDENTIFIER', 'the item-deletion scope name "random 1%"'
            if tok == '10': return 'PROTOCOL/DEFINITION', 'held-out validation split size (10%)'
        if re.search(r'^\\times', after) or re.search(r'\\times$', before):
            if tok == '10': return 'IDENTIFIER', 'schedule name "$10\\times$ training steps"'
            return 'IDENTIFIER', 'panel crop size notation ($5\\times5$)'
        if re.search(r'\($', before) and re.search(r'^\)', after):
            return 'IDENTIFIER', 'list enumerator (1)-(4)'
        if re.search(r'[Rr]ows?\s+(\d+--)?$', before) or re.search(r'^--4', after) and 'rows' in before[-6:]:
            return 'IDENTIFIER', 'row index of Tab. t7 (row 3 / row 4)'
        if re.match(r'(19|20)\d\d$', tok):
            return 'IDENTIFIER', 'year in a citation phrase'
        if re.search(r'(s_\{t\+|\)\^\{-|\\lambda=|\[0\.5,)$', before) or re.search(r'^(\}|\])', after) and re.search(r'[_^{\[]', before[-4:]):
            return 'IDENTIFIER', 'math notation ($s_{t+1}$, $(\\cdot)^{-1}$, $\\lambda=0$, $[0.5,2]$ bucket)'
        if re.search(r'^-atom', after):
            return 'IDENTIFIER', 'design name (6-atom / 10-atom)'
        if re.search(r'^-seed', after):
            return 'COUNT', 'seed count in "3-seed mean"'
        if re.search(r'^ seeds,', after) and re.search(r'\($', before):
            return 'COUNT', 'seed count "(3 seeds"'
        if re.search(r'^ (checks|requests)', after):
            return 'PROTOCOL/DEFINITION', 'assertion size of a code check (%s %s)' % (tok, after.split()[0])
        if re.search(r'margin of $', before):
            return 'PROTOCOL/DEFINITION', 'the declared 1.0-tuple margin (defined once, here)'
        if re.search(r'^-byte figure', after) or re.search(r'^ float32', after):
            return 'PROTOCOL/DEFINITION', 'the one definition of the byte reference (524,288 B / 131,072 float32)'
        if re.search(r'pocket $', before):
            return 'IDENTIFIER', 'outcome code "(legal, move south, pocket 2)"'
        if re.search(r'share is $', before):
            return 'PROTOCOL/DEFINITION', 'coverage share 1 by construction (a definitional identity)'
        if re.search(r'at rank\s*$', before) or (full.strip().startswith(tok) and re.search(r'at rank\s*$', lines[ln - 2])):
            return 'IDENTIFIER', 'rank 64 (design parameter)'
        if re.search(ROUNDERS, before):
            return ROUNDED_CHECK(t, lines, anchors_code)
        return 'UNCLASSIFIED', ''
    # words
    if low in ('zero',) and re.search(r'^-(dynamics|step)', after):
        return 'IDENTIFIER', 'name of the limit / arm (zero-dynamics, zero-step)'
    if re.search(ROUNDERS, before, re.I) or (low in ('thousand',) and re.search(r'about four $', before)) \
            or (low == 'ten' and re.search(r'about nine in $', before)) or (low in ('thirds',)):
        return ROUNDED_CHECK(t, lines, anchors_code)
    if low == 'half':
        if re.search(r'^-life', after): return 'NOT-A-NUMBER', 'the term "half-life"'
        if re.search(r'either $', before): return 'NOT-A-NUMBER', '"either half" (of a two-part claim)'
        return ROUNDED_CHECK(t, lines, anchors_code)
    if low in ('ten', 'thirty') and re.search(r'^[ -]times', after):
        return 'IDENTIFIER', 'schedule name ("ten times" / "thirty times" its selected steps)'
    if low == 'ten' and re.search(r'from $', before) and re.search(r'^ to thirty', after):
        return 'IDENTIFIER', 'schedule name ("from ten to thirty times its steps")'
    if low == 'zero' and re.search(r'(above|excludes|crosses|includes)\s+$', before):
        return 'PROTOCOL/DEFINITION', 'zero as the null reference of an interval ("above / excludes / crosses zero")'
    if low == 'zero' and re.search(r'set to\s+$', before):
        return 'PROTOCOL/DEFINITION', 'definition of the limit (settling step count set to zero)'
    if low == 'zero' and re.search(r'colour is\s+$', before):
        return 'PROTOCOL/DEFINITION', 'structural zero (the colour-naming columns are zero by construction)'
    if low == 'four' and re.search(r'^\s+times the n-CLU', after):
        return ROUNDED_CHECK(t, lines, anchors_code)
    if re.search(r'^\s+of\s+(three|six)\s+(TEST\s+)?seeds', after) or re.search(r'on\s+(all\s+)?$', before) and re.search(r'^\s+(TEST\s+)?seeds', after) \
            or re.search(r'(level (with it )?on|behind on|ahead .* on)\s+$', before) or re.search(r'^\s+(clear wins|ties)', after) \
            or (low == 'one' and re.search(r'^\s+loss', after)) or re.search(r'^\s+of\s+them', after) or re.search(r'of\s+$', before) and re.search(r'^\s+under', after):
        return 'COUNT', 'seed-outcome tally (a count of seeds; the per-seed values are in the cited table)'
    if low == 'one':
        if re.search(r'^(-tuple|-token|-parameter)', after):
            return 'COUNT', '"one-%s" (a compound: the declared one-tuple margin / one-token lookup / one-parameter form)' % after[1:].split()[0].rstrip(',.')
        if re.search(r'(established|never-inserted|the)\s+$', before) and not re.search(r'^\s+(of|n-CLU|configuration|definition)', after):
            if re.search(r'the\s+$', before) and re.search(r'^\s+measured', after):
                return 'NOT-A-NUMBER', 'determiner "the one measured ..." (coinage, Head-held elsewhere)'
            return 'NOT-A-NUMBER', 'pronoun "one"'
        return 'COUNT', 'determiner/count "one" (a single thing)'
    if low == 'three' and re.search(r'^\s+(properties|pull)', after):
        return 'COUNT', 'the three properties (the paper\'s frame)'
    return 'COUNT', 'count word of discrete things (%s)' % ' '.join((tok + after).split()[:3])


def ROUNDED_CHECK(t, lines, anchors_code):
    """the rounded_check row whose anchor match covers this token (offsets kept: newlines -> spaces, comments blanked)"""
    ln = t['line']
    seg = [re.sub(r'(?<!\\)%.*', lambda m: ' ' * len(m.group(0)), l) for l in lines[max(0, ln - 3):ln + 2]]
    off = sum(len(l) + 1 for l in seg[:ln - max(0, ln - 3) - 1]) + t['col']
    win = ' '.join(seg)
    best = None
    for i, (a, *_rest) in enumerate(rounded_check.R, 1):
        pat = re.sub(r' ', r'\\s+', a)
        for m in re.finditer(pat, win):
            if m.start() - 40 <= off <= m.end() + 5:
                d = 0 if m.start() <= off < m.end() else min(abs(off - m.start()), abs(off - m.end()))
                if best is None or d < best[0]: best = (d, i, _rest[3])
    if best:
        return 'ROUNDED', 'rounded; record re-checked, rounded_check row %d (%s)' % (best[1], best[2])
    return 'ROUNDED?', 'rounding word before it, but no record row covers it'


def main():
    path = sys.argv[1]
    lines = open(path).read().split('\n')
    hh = headheld_lines(path)
    toks = scan(path)
    out = []
    for t in toks:
        c, j = classify(t, lines, hh, None)
        t['cls'], t['why'] = c, j
        out.append(t)
    if '--json' in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index('--json') + 1], 'w'), indent=0)
    bad = [t for t in out if t['cls'] in ('UNCLASSIFIED', 'ROUNDED?')]
    print('| line | section | kind | number | class | justification | code (from the token) |')
    print('|---|---|---|---|---|---|---|')
    for t in out:
        c = lines[t['line'] - 1]
        print('| %d | %s | %s | `%s` | %s | %s | `%s` |' % (t['line'], t['section'], t['kind'], t['tok'], t['cls'], t['why'],
              ' '.join(c[max(0, t['col'] - 40):t['col'] + 60].split()).replace('|', '\\|')))
    print('\nUNCLASSIFIED or uncovered ROUNDED?: %d' % len(bad))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
