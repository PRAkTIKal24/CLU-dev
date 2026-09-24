#!/usr/bin/env python3
"""Prove no retired number survives in RENDERED text. usage: retired_grep.py <pdf> [...]
Extracts text with Ghostscript txtwrite, tokenises every number as a whole token, and reports exact
matches against the retired set. Exit 1 on any hit, 2 on failed extraction."""
import re, subprocess, sys, tempfile
RETIRED = {
 # (M) the +39...+49 margins over the 16,384-row comparators, the range endpoints, the floor margins
 'margins': '42.303 44.165 40.945 42.734 42.771 39.716 46.266 46.752 46.046 47.468 47.376 49.064 46.725 46.743 43.450 43.092 39.266 42.330 39.367 39.7 49.1 41.74 42.47',
 # (C) the 16,384-row comparators' scores; the step-matched diagnostic; the old strongest-rival triple
 'comparators': '2.9333 3.4333 5.2667 7.9592 11.7347 2.5000 2.7000 4.9167 7.5306 14.8980 0.6667 1.7167 1.9184 0.6327 4.7347 0.0167 0.1020 0.3000 1.7333 1.1429 3.1633 3.0833 4.3667 5.0204 8.0612 5.535 6.266 1.651 0.037 5.193 3.239 8.174 4.761 9.404 1.229 0.651 3.073 5.5 38.606 36.358 37.028 42.431 39.067 37.550 37.009 42.642 38.6 39.1 39.2 1,280 30.0275 28.9174 37.5138',
 # (D) the 16,384-row deletion comparators and approximate unlearners
 'deletion16k': '8.86 8.87 34.62 15.99 15,032 15,717 16,220 1,352 164 46.899 48.360 45.910 4.022 2.169 6.697 0.03661 0.03817 0.06057 0.04111 0.51518 1.26223 9.4 20.8 296,960',
 # (L) learned_ref_ownpool (2,560-step) scores and learned_1x
 'ownpool_1x': '5.5667 7.5833 9.3000 9.1835 11.1193 15.1284 3.117 536,270 536,972 1,029,776 3,131,564 3,260,030 4.356',
}
WORDS = ['NegGrad', 'learned_1x', '16,384']   # 16,384 checked separately: only the byte residual '16,384 B' may remain
NUM = re.compile(r'(?<![\d.,])(\d[\d,]*(?:\.\d+)?)(?![\d])')
rc = 0
for pdf in sys.argv[1:]:
    with tempfile.NamedTemporaryFile(suffix='.txt') as f:
        subprocess.run(['gs', '-q', '-dNOPAUSE', '-dBATCH', '-sDEVICE=txtwrite', '-sOutputFile=' + f.name, pdf], check=True, capture_output=True)
        lines = open(f.name, errors='replace').read().split('\n')
    nwords = sum(len(l.split()) for l in lines)
    print('#### %s  (extracted words: %d)' % (pdf, nwords))
    if nwords < 5000:
        print('EXTRACTION FAILED'); sys.exit(2)
    for cls, s in RETIRED.items():
        bad = set(s.split()); hits = []
        for i, l in enumerate(lines):
            for m in NUM.finditer(l.replace('−', '-')):
                t = m.group(1).rstrip('.,')
                if t in bad:
                    hits.append((i + 1, t, l.strip()[:150]))
        print('== %s: %d hit(s)' % (cls, len(hits)))
        for h in hits[:40]:
            print('   line %d  token %s  | %s' % h)
        rc |= bool(hits)
    for w in WORDS:
        hits = [(i + 1, l.strip()[:150]) for i, l in enumerate(lines) if w in l]
        if w == '16,384':
            hits = [h for h in hits if not re.search(r'16,384 B\b', h[1])]
        print('== word %r: %d hit(s)' % (w, len(hits)))
        for h in hits[:20]:
            print('   line %d | %s' % h)
        rc |= bool(hits)
    # the '+39 ... +49' range written in prose
    hits = [(i + 1, l.strip()[:150]) for i, l in enumerate(lines) if re.search(r'\+\s?39\b(?![.\d])|\+\s?49\b(?![.\d])', l)]
    print('== prose +39/+49: %d hit(s)' % len(hits)); rc |= bool(hits)
    for h in hits[:10]:
        print('   line %d | %s' % h)
sys.exit(1 if rc else 0)
