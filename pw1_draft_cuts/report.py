#!/usr/bin/env python3
"""Evidence tables for pw1-draft-cuts. usage: report.py SNAPDIR STAGEDIR BUILD_BEFORE BUILD_AFTER OUT.md

Removal table (with the large spans broken down by heading / float), every \\ref removed or rewritten,
PLACEHOLDER / TODO lists before and after, the section / figure / table index from the .aux files,
page counts (whole draft and per top-level section), and the residual scans (source and PDF text)."""
import json, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'pw1_draft_sweeps'))
from texnum import macro_args, split_comment

SNAP, STAGE, BB, BA, OUT = sys.argv[1:6]
DR = [('v2', 'DRAFT-v2'), ('style', 'DRAFT-v2-style')]
out = []
P = out.append


def code_only(s):
    return '\n'.join(split_comment(l)[0] for l in s.split('\n'))


def words(s):
    c = code_only(s)
    c = re.sub(r'\\(label|ref|eqref|cite[pt]?|nocite|begin|end|setlength|figslot)\{[^}]*\}', ' ', c)
    c = re.sub(r'\\[A-Za-z]+\*?', ' ', c)
    c = re.sub(r'[{}$&\\~^_]', ' ', c)
    return len(re.findall(r'[A-Za-z0-9][A-Za-z0-9\'.,/:+\-]*', c))


def one(s, n=90):
    s = re.sub(r'\s+', ' ', s).strip().replace('|', '\\|')
    return s[:n] + ('…' if len(s) > n else '')


START = re.compile(r'^(\\section|\\subsection|\\paragraph|\\begin\{table\}|\\figslot|\\item )')


def pieces(block, first_line):
    """Split a removed span at headings / floats / list items; each piece: (label-or-heading, l0, l1, words)."""
    lines = block.split('\n')
    cuts = [0] + [i for i, l in enumerate(lines) if i and START.match(l)]
    res = []
    for a, b in zip(cuts, cuts[1:] + [len(lines)]):
        chunk = '\n'.join(lines[a:b])
        if not code_only(chunk).strip():
            continue
        labs = re.findall(r'\\label\{([^}]*)\}', code_only(chunk))
        head = re.search(r'\\(section|subsection|paragraph)\*?\{([^}]*)\}', chunk)
        fs = re.search(r'\\figslot\{([^}]*)\}', chunk)
        name = ' '.join('`%s`' % x for x in labs) or ('%s{%s}' % (head.group(1), head.group(2)) if head else '') \
            or ('figslot `%s`' % fs.group(1) if fs else '') or one(code_only(chunk), 60)
        if head and labs:
            name = '%s{%s} %s' % (head.group(1), one(head.group(2), 50), name)
        elif head:
            name = '%s{%s}' % (head.group(1), one(head.group(2), 70))
        last = b - 1
        while last > a and not lines[last].strip():
            last -= 1
        res.append((name, first_line + a, first_line + last, words(chunk)))
    return res


# ---------------------------------------------------------------- 1. removal table
P('## 1. Removal table (every block removed or rewritten, per draft)')
P('')
P('Line ranges are in the snapshot drafts (`snap-before/`). Words = rendered-word estimate (comments, control words and '
  'braces dropped). `kind`: remove (text gone, pointer comment left), replace (rewritten), move (relocated, not removed).')
P('')
reps = {w: json.load(open(os.path.join(STAGE, 'CUTS-%s.json' % w))) for w, _ in DR}
ops = {w: reps[w]['ops'] for w, _ in DR}
SNAPTXT = {w: open(os.path.join(SNAP, n + '.tex')).read() for w, n in DR}


def snapspan(o, w):
    """(first, last) snapshot line of an op's `before` text; a block that was edited earlier in the run
    (the read-verb appendix after the conditions figure left it) is located by its first and last 150 chars."""
    t, b = SNAPTXT[w], o['before']
    if t.count(b) == 1:
        i = t.index(b)
        return t.count('\n', 0, i) + 1, t.count('\n', 0, i + len(b.rstrip('\n'))) + 1
    h, e = b[:150], b.rstrip('\n')[-150:]
    if t.count(h) == 1 and t.count(e) == 1:
        i, j = t.index(h), t.index(e) + len(e)
        return t.count('\n', 0, i) + 1, t.count('\n', 0, j) + 1
    return None


def snaplines(o, w):
    r = snapspan(o, w)
    return '%d–%d' % r if r else '(staged text only)'


# pair the ops of the two drafts by (item, what) in order
P('| # | item | kind | what | v2 lines | v2 words −/+ | style lines | style words −/+ | refs removed |')
P('|---|---|---|---|---|---|---|---|---|')
for k, (a, b) in enumerate(zip(ops['v2'], ops['style']), 1):
    assert a['item'] == b['item'] and a['what'] == b['what'], (a['what'], b['what'])
    lr = lambda o, w: snaplines(o, w)
    P('| %d | %s | %s | %s | %s | %d / %d | %s | %d / %d | %s |' % (
        k, a['item'], a['kind'], one(a['what'], 140), lr(a, 'v2'), a['words_removed'], a['words_added'], lr(b, 'style'),
        b['words_removed'], b['words_added'], ('—' if a['kind'] == 'move' else ', '.join('`%s`' % r for r in a['refs_removed']) or '—')))
P('')
tot = {w: (sum(o['words_removed'] for o in ops[w] if o['kind'] != 'move'), sum(o['words_added'] for o in ops[w] if o['kind'] != 'move')) for w, _ in DR}
P('Totals (moves excluded): v2 %d words removed, %d written; style %d removed, %d written.' % (tot['v2'] + tot['style']))
P('')
P('### 1a. The large spans, broken down (snapshot line ranges)')
P('')
for w, name in DR:
    P('**%s**' % name)
    P('')
    P('| op | block (label / heading / float) | lines | words |')
    P('|---|---|---|---|')
    for k, o in enumerate(ops[w], 1):
        if o['kind'] == 'move':
            continue
        r = snapspan(o, w)
        if not r or r[1] - r[0] < 12:
            continue
        blk = o['before'] if SNAPTXT[w].count(o['before']) == 1 else '\n'.join(SNAPTXT[w].split('\n')[r[0] - 1:r[1]])
        for nm, l0, l1, wd in pieces(blk, r[0]):
            P('| %d | %s | %d–%d | %d |' % (k, nm, l0, l1, wd))
    P('')

# ---------------------------------------------------------------- 2. refs
P('## 2. Every `\\ref` removed or rewritten')
P('')


def refs_with_line(t):
    res = []
    for i, l in enumerate(t.split('\n'), 1):
        for m in re.finditer(r'\\ref\{([^}]*)\}', split_comment(l)[0]):
            res.append((m.group(1), i))
    return res


for w, name in DR:
    b = open(os.path.join(SNAP, name + '.tex')).read()
    a = open(os.path.join(STAGE, name + '.tex')).read()
    rb, ra = refs_with_line(b), refs_with_line(a)
    from collections import Counter
    cb, ca = Counter(r for r, _ in rb), Counter(r for r, _ in ra)
    lost, added = cb - ca, ca - cb
    P('**%s**: `\\ref` tokens %d → %d. Removed (key × count): %s. Added: %s.' % (
        name, len(rb), len(ra), ', '.join('`%s`×%d' % kv for kv in sorted(lost.items())) or 'none',
        ', '.join('`%s`×%d' % kv for kv in sorted(added.items())) or 'none'))
    labs = set(re.findall(r'\\label\{([^}]*)\}', code_only(a))) | set(re.findall(r'\}\{(fig:[^}]+)\}', code_only(a)))
    dead = sorted(set(r for r, _ in ra) - labs)
    P('Keys referenced but not labelled after: %s.' % (dead or 'none'))
    P('')

# ---------------------------------------------------------------- 3. placeholders / todos
P('## 3. `\\PLACEHOLDER` and `\\TODO`, before → after (document order)')
P('')
for w, name in DR:
    b = open(os.path.join(SNAP, name + '.tex')).read()
    a = open(os.path.join(STAGE, name + '.tex')).read()
    for mac in ('PLACEHOLDER', 'TODO'):
        xb, xa = macro_args(b, mac), macro_args(a, mac)
        P('**%s `\\%s`**: %d → %d; the surviving ones byte-identical and in order: **%s**.' % (
            name, mac, len(xb), len(xa), [x for x in xb if x in xa] == xa))
        for i, x in enumerate(xb, 1):
            P('%d. %s `%s`' % (i, 'kept' if x in xa else '**REMOVED**', one(x, 150)))
        P('')

# ---------------------------------------------------------------- 4. index
P('## 4. Section, figure and table index (from the .aux of each build)')
P('')


def auxlabels(path):
    res = []
    for l in open(path, errors='replace'):
        if not l.startswith('\\newlabel{'):
            continue
        key = l[10:l.index('}')]
        s = l[l.index('}') + 1:]
        # {{num}{page}{title}{anchor}{}}
        args, depth, cur = [], 0, ''
        for ch in s:
            if ch == '{':
                depth += 1
                if depth == 2:
                    cur = ''
                    continue
            elif ch == '}':
                depth -= 1
                if depth == 1:
                    args.append(cur)
                    continue
            if depth >= 2:
                cur += ch
        if len(args) >= 4 and not key.endswith('@cref'):
            res.append((key, args[0], args[1], args[2], args[3]))
    return res


for w, name in DR:
    for tag, bd in (('before', BB), ('after', BA)):
        L = auxlabels(os.path.join(bd, name + '.aux'))
        P('<details><summary><b>%s, %s</b>: %d labels</summary>' % (name, tag, len(L)))
        P('')
        P('| kind | number | label | page | title / caption first words |')
        P('|---|---|---|---|---|')
        for key, num, page, title, anchor in L:
            kind = anchor.split('.')[0]
            if kind not in ('section', 'subsection', 'figure', 'table', 'appendix'):
                continue
            P('| %s | %s | `%s` | %s | %s |' % (kind, num, key, page, one(re.sub(r'\\textbf \{([^}]*)\}', r'\1', title), 80)))
        P('')
        P('</details>')
        P('')

# compact figure / table number maps, before → after
P('### 4a. Figure and table numbers, before → after (DRAFT-v2)')
P('')
Lb = {k: (n, p) for k, n, p, t, a in auxlabels(os.path.join(BB, 'DRAFT-v2.aux')) if a.split('.')[0] in ('figure', 'table')}
La = {k: (n, p) for k, n, p, t, a in auxlabels(os.path.join(BA, 'DRAFT-v2.aux')) if a.split('.')[0] in ('figure', 'table')}
P('| label | before (number, page) | after (number, page) |')
P('|---|---|---|')
for k in sorted(set(Lb) | set(La), key=lambda k: (k.split(':')[0], float(Lb.get(k, La.get(k))[0]) if Lb.get(k, La.get(k))[0].replace('.', '', 1).isdigit() else 0)):
    P('| `%s` | %s | %s |' % (k, 'Fig./Tab. %s, p. %s' % Lb[k] if k in Lb else '—', 'Fig./Tab. %s, p. %s' % La[k] if k in La else '**removed**'))
P('')

# ---------------------------------------------------------------- 5. pages
P('## 5. Page counts')
P('')


def pages(bd, name):
    m = re.search(r'Output written on .*?\((\d+) pages', open(os.path.join(bd, name + '.log'), errors='replace').read(), re.S)
    return int(m.group(1))


def secpages(bd, name):
    L = [(n, int(p), t) for k, n, p, t, a in auxlabels(os.path.join(bd, name + '.aux')) if a.startswith('section.') or a.startswith('appendix.')]
    return L


P('| draft | pages before | pages after |')
P('|---|---|---|')
for w, name in DR:
    P('| %s | %d | %d |' % (name, pages(BB, name), pages(BA, name)))
P('')
for w, name in DR:
    for tag, bd in (('before', BB), ('after', BA)):
        L = secpages(bd, name)
        tot = pages(bd, name)
        P('**%s, %s** (top-level sections: number, start page, span in pages up to the next top-level section; the last runs to the end):' % (name, tag))
        P('')
        P('| § | title | start p. | pages |')
        P('|---|---|---|---|')
        for i, (n, p, t) in enumerate(L):
            nxt = L[i + 1][1] if i + 1 < len(L) else tot + 1
            P('| %s | %s | %d | %d |' % (n, one(t, 70), p, max(nxt - p, 0)))
        P('')

open(OUT, 'w').write('\n'.join(out) + '\n')
print('wrote', OUT, len(out), 'lines')
