"""ROUNDED check for part 7 (item 5 on the Discussion limitations list): every rounded or relative phrase now in the
list, with the recorded value strings that must occur verbatim in the record (evidence root) and the rule the phrase
asserts, evaluated on the recorded numbers. Rules as pw2_draft_numbers/rounded.py: never round up past the record
("about X" = X at or below every record it summarises; "nearly X" = X above the record and within 10%; comparisons exact).
usage: python3 rounded.py [--plant]  -> markdown table; exit 1 on any FAIL. --plant adds a rounding past its record (must FAIL)."""
import os, sys
EV = os.path.expanduser('~/clu/evidence/outputs/')
def txt(f):
    t = open(EV + f, errors='replace').read().replace('−', '-')
    return t + '\n' + t.replace(',', '')

R = []
def row(phrase, where, checks, ok, why):
    R.append((phrase, where, checks, ok, why))

F = 'w20-shapes-data-matched/FINDINGS.md'; T = 'w13-seeds-and-tables/TABLES.md'
row('about six tuples per held-out combination (red sphere, survivors)', 'Limitation 2', [(F, ['6.13', '6.43', '6.30'])],
    all(6 <= v < 6.5 for v in (6.13, 6.43, 6.30)), 'each seed 6.13 / 6.43 / 6.30 rounds to 6 and is >= 6')
row('about thirty on the combinations that request red with another shape at seed 0', 'Limitation 2', [(F, ['32.1'])],
    30 <= 32.1 < 35, '-32.1 per combination (20 combinations, seed 0); 30 <= 32.1')
row('at most about a tenth of a tuple (random 1% deletion)', 'Limitation 2', [(F, ['-0.02 / -0.11 / -0.01'])],
    0.1 <= max(0.02, 0.11, 0.01) < 0.15, 'item deletion moves -0.02 / -0.11 / -0.01; largest 0.11 rounds to 0.1 (same phrase and rule as Sec. 4.3, pw2-draft-numbers ROUNDED row 26)')
row('nearly a gigabyte (the book\'s retained statistic)', 'Limitation 3', [('w20-shapes-data-matched/SHAPES-MATCHED.json', ['967304800']), (F, ['0.97 GB'])],
    0.9 <= 967304800 / 1e9 < 1.0, '967,304,800 B = 0.967 GB (decimal): below 1 and within 10%')
row('its refit takes longer than a baseline\'s retrain-without', 'Limitation 3',
    [(F, ['2,367']), ('w20-shapes-data-matched/SHAPES-MATCHED.md', ['2064.5'])],
    2367 > 2064.5, 'fastest book refit 2,367 s (2,367.3 in Tab. del) > slowest baseline retrain 2,064.5 s (Tab. rivals-del)')
row('about four times the reference (the deletable object at K*)', 'Limitation 4', [('w13-referee-2/REFEREE-v1.md', ['4.0897', '4.0768', '4.0847'])],
    4 <= 4.0768 < 4.5, 'per seed 4.0897 / 4.0768 / 4.0847 (Tab. ledger-kstar, cited to REFEREE-v1.md; TABLES.md prints the seed-0 4.0897 only) all round to 4 and are >= 4')
row('U-shaped, with its minimum at K=24', 'Limitation 4', [(T, ['3.9942', '4.0877'])],
    min([(7.8545, 5), (5.6450, 8), (4.9720, 10), (4.5656, 12), (4.1529, 16), (4.0069, 20), (3.9942, 24), (4.0567, 28), (4.0877, 32)])[1] == 24,
    'four-part x column of Tab. kcurve: 7.8545 … 4.0069 (K=20), 3.9942 (K=24), 4.0567 (K=28), 4.0877 (K=32); minimum at K=24')
row('on the one configuration at K>=36 whose composition was measured, composition collapses', 'Limitation 4',
    [(T, ['0.991x', '0.0648', '0.8918']), ('w13-referee-2/REFEREE-v1.md', ['0.991'])],
    0.0648 < 0.1 * 0.8918, 'K=40 / A=7 at 0.991x the pin: valid mean 0.0648, against 0.8918 at K=5 (TABLES.md, the w12r qualifier kept verbatim in sense)')
row('the MLP at seed 0 still gains more than a tuple from ten to thirty times its steps', 'Limitation 5', [(F, ['+1.47'])],
    1.47 > 1, '+1.47 (36.71 -> 40.01 -> 41.48 at seed 0)')
if '--plant' in sys.argv:
    row('PLANTED: about four times the reference (record 3.9942)', '-', [(T, ['3.9942'])], 4 <= 3.9942, 'planted: 4 > 3.9942, rounds up past the record')

fails = 0
print('| # | phrase in the text | where | record file(s) | verbatim in record | rule holds | recorded values / rule |')
print('|---|---|---|---|---|---|---|')
for i, (ph, wh, checks, ok, why) in enumerate(R, 1):
    found, allf = [], True
    for f, nd in checks:
        t = txt(f)
        for n in nd:
            hit = n in t or n.replace(',', '') in t
            allf &= hit
            found.append(('' if hit else 'MISSING ') + n)
    good = ok and allf
    fails += not good
    print('| %d | %s | %s | %s | %s | %s | %s |' % (i, ph, wh, '; '.join('`%s`' % f for f, _ in checks),
          ('yes' if allf else 'NO') + ' (' + ', '.join(found) + ')', 'PASS' if good else 'FAIL', why))
print('\n%d rows, %d FAIL' % (len(R), fails))
sys.exit(1 if fails else 0)
