"""Every rounded or relative number written into the main text by PW-2 item 5, checked against its record.
For each: the phrase, the recorded value string(s) that must occur verbatim in the file (evidence root), and the
rule the phrase asserts, evaluated on the recorded numbers (never rounds up past the record; "nearly X" = X above
the record and within 10%; "about X" = X at or below every record it summarises, rounding to X; comparisons exact).
usage: python3 rounded.py [--plant]  -> prints the table (markdown); exit 1 on any FAIL. --plant flips one rule (positive control)."""
import os, sys
EV = os.path.expanduser('~/clu/evidence/outputs/')
OUT = os.path.expanduser('~/clu/outputs/')
def txt(f):
    p = f if f.startswith('/') else (OUT + f[len('../../outputs/'):] if f.startswith('../../outputs/') else EV + f)
    t = open(p, errors='replace').read().replace('−', '-')
    return t + '\n' + t.replace(',', '')

R = []
def row(phrase, where, file, needles, ok, why):
    R.append((phrase, where, file, needles, ok, why))

row('about seven (more tuples than the strongest, an MLP)', 'Abstract; Contribution 1; Summary (pre-existing)', 'w20-shapes-data-matched/SHAPES-MATCHED.md',
    ['+7.034'], 6.5 <= 7.034 < 7.5 and 7 <= 7.034, 'mean paired margin +7.034 rounds to 7; 7 <= 7.034')
row('by more (over the transformer)', 'Contribution 1', 'w20-shapes-data-matched/FINDINGS.md', ['+7.03'],
    min(21.936, 19.615, 10.468) > 7.034, 'transformer paired margins 21.936 / 19.615 / 10.468 (Tab. t7) all exceed 7.034')
row('about six (survivors fall, red sphere)', 'Abstract; Sec. 4.3', 'w20-shapes-data-matched/FINDINGS.md', ['6.13', '6.43', '6.30'],
    all(6 <= v < 6.5 for v in (6.13, 6.43, 6.30)), 'each seed 6.13 / 6.43 / 6.30 rounds to 6 and is >= 6')
row('far more than after a random item deletion', 'Abstract', 'w20-shapes-data-matched/FINDINGS.md', ['6.13', '0.11'],
    6.13 > 10 * 0.11, 'smallest conjunction loss 6.13 against largest item move 0.11')
row('as large as its fit data / the size of its fit data', 'Abstract (Contribution 2 pre-existing)', 'w20-shapes-data-matched/FINDINGS.md', ['0.97 GB', 'the fit data itself'],
    True, 'record: "first moment of its 117,964 fit rows: 0.97 GB, i.e. the fit data itself"')
row('nearly a gigabyte', 'Contribution 2; Sec. 4.3', 'w20-shapes-data-matched/SHAPES-MATCHED.json', ['967304800'],
    0.9 <= 967304800 / 1e9 < 1.0, '967,304,800 B = 0.967 GB (decimal), below 1 and within 10%')
row('refit slower than / longer than retraining a baseline', 'Abstract; Contribution 2; Sec. 4.4', 'w20-shapes-data-matched/FINDINGS.md', ['2,367'],
    2367 > 2064.5, 'fastest book refit 2,367 s (FINDINGS) > slowest baseline retrain 2,064.5 s (SHAPES-MATCHED.md, Tab. rivals-del)')
row('(slowest retrain) 2,064.5 s', 'Sec. 4.4 comparison', 'w20-shapes-data-matched/SHAPES-MATCHED.md', ['2064.5'], True, 'printed in the record')
row('several times the byte reference', 'Abstract', 'w13-seeds-and-tables/TABLES.md', ['4.0768', '4.0897'],
    3 <= 4.0768, '4.0768 to 4.0897 times')
row('about four thousand unique works', 'Sec. 1.1', '_thinkpad/w8/w8-novelty/NOVELTY.md', ['4,050'],
    4000 <= 4050 < 4500, 'approximately 4,050 rounds to 4,000 and 4,000 <= 4,050')
row('ten generation-side works; most of them false positives', 'Sec. 1.1', '_thinkpad/w8/w8-novelty/NOVELTY.md', ['10'],
    7 / 10 > 0.5, '10 works, 7 false positives (as printed in the draft before; record NOVELTY.md)')
row('four (claim atoms) of the six', 'Sec. 1.1', '_thinkpad/w8/w8-novelty/NOVELTY.md', ['4 of 6', '4/6'],
    True, 'exact count')
row('about nine in ten (n-CLU valid, seed 0)', 'Sec. 4.1', '../../outputs/pw1-render-shapes/REQUESTS-pw1.json', ['5177', '5688'],
    0.9 <= 5177 / 5688 < 0.95, '5,177 / 5,688 = 0.910')
row('more often than any baseline; lookup table least often', 'Sec. 4.1', '../../outputs/pw1-render-shapes/REQUESTS-pw1.json', ['4361', '3703', '2786', '3110', '897'],
    5177 > max(4361, 3703, 2786, 3110) and 897 < min(4361, 3703, 2786, 3110, 5177), 'n-CLU 5,177 > MLP 4,361 > …; table 897 lowest')
row('almost always wrong in one factor only', 'Sec. 4.1 (pre-existing words)', '../../outputs/pw1-render-shapes/factor-miss-seed0.txt', ['497', '511'],
    497 / 511 > 0.95, '497 / 511 = 0.973')
row('not confined to the capsule', 'Sec. 4.1', '../../outputs/pw1-render-shapes/REQUESTS-pw1.json', ['170', '310'],
    170 < 310, '170 of 310 are capsules')
row('tracks that count closely, inversely', 'Sec. 4.2', 'w12-composition-ladder/FINDINGS.md', ['0.96', '0.98'],
    True, 'r = -0.96 to -0.98 (record prints both bounds)')
row('about one tuple (LSTM spot check)', 'Sec. 4.2', 'w20-shapes-data-matched/FINDINGS.md', ['20.55', '21.85'],
    1 <= 21.85 - 20.55 < 1.5, '21.85 - 20.55 = 1.30')
row('four times the n-CLU\'s content rows', 'Sec. 4.2 (pre-existing words)', 'w21-ladder-rivals-32k/CAPTION.md', ['32,768'],
    131072 / 32768 == 4, '131,072 / 32,768 = 4 exactly')
row('up to about four novel tuples; most rung-seed legs', 'Sec. 4.2', 'w21-ladder-rivals-32k/FINDINGS.md', ['4.13', '12 of 15', '12/15'],
    4 <= 4.13 < 4.5 and 12 / 15 > 0.5, 'largest |difference| 4.13 (range -2.15 to +4.13); 12 of 15')
row('nearly five tuples (TEST rise n=3 -> 6)', 'Sec. 4.2', 'w15-coverage-control/FINDINGS.md', ['4.989'],
    4.5 <= 4.989 < 5, '+4.989 is below 5, within 10%')
row('coverage share complete; mask channel contributes nothing', 'Sec. 4.2', 'w15-coverage-control/FINDINGS.md', ['1.00', '0.000'],
    True, 'coverage share 1.00 on every seed; mask channel 0.000')
row('about thirty each (red with another shape, seed 0)', 'Sec. 4.3', 'w20-shapes-data-matched/FINDINGS.md', ['32.1'],
    30 <= 32.1 < 35, '-32.1 per combination; 30 <= 32.1')
row('about one (sphere in another colour, seed 0)', 'Sec. 4.3', 'w20-shapes-data-matched/FINDINGS.md', ['1.47'],
    1 <= 1.47 < 1.5, '-1.47 rounds to 1; 1 <= 1.47')
row('the others barely move', 'Sec. 4.3', 'w20-shapes-data-matched/FINDINGS.md', ['+0.51'],
    abs(0.51) < 1, '+0.51, inside the one-tuple margin')
row('at most about a tenth of a tuple (random 1% deletion)', 'Sec. 4.3', 'w20-shapes-data-matched/FINDINGS.md', ['0.11'],
    0.1 <= 0.11 < 0.15, 'largest item move 0.11 (0.110) rounds to 0.1 (stated as "about")')
row('less than the declared one-tuple margin (value deletion)', 'Sec. 4.3', 'w20-shapes-data-matched/FINDINGS.md', ['0.61'],
    max(0.06, 0.33, 0.61) < 1.0, '-0.06 / +0.33 / -0.61, all |.| < 1.0')
row('about 0.6 relative RMS; as much on the others', 'Sec. 4.3', 'w13-seeds-and-tables/TABLES.md', ['0.610083', '0.614277'],
    0.6 <= 0.610083 < 0.65 and 0.614277 >= 0.610083, '0.610083 (red) and 0.614277 (others)')
row('only at roundoff (with the carry)', 'Sec. 4.3', 'w13-seeds-and-tables/TABLES.md', ['6.194', '6.163'],
    True, '6.194e-08 and 6.163e-08 relative')
row('nearly all of the address book\'s bytes', 'Sec. 4.3', 'w13-seeds-and-tables/TABLES.md', ['1015808', '1060864'],
    0.9 <= 1015808 / 1060864 < 1, '1,015,808 / 1,060,864 = 0.958')
row('far lower minimum cosine', 'Sec. 4.3', 'w13-seeds-and-tables/TABLES.md', ['0.1999', '0.9976'],
    0.1999 < 0.25 * 0.9976, 'cos_min 0.1999 against 0.9976')
row('a byte each; no measurable time', 'Sec. 4.5', 'w13-seeds-and-tables/TABLES.md', ['0.0000'],
    True, 'two holes cost 2 bytes and take 0.0000 s')
row('a free value-presence bitmap of a few bytes; perfectly', 'Sec. 4.5', 'w12-deletion-baselines/FINDINGS.md', ['1.0000'],
    True, '8-byte bitmap; AUROC / refusal 1.0000')
row('beyond a declared one-tuple bar on two of them', 'Sec. 4.5', 'w13-seeds-and-tables/TABLES.md', ['2.0899', '0.9101', '6.7416'],
    sum(v > 1.0 for v in (2.0899, 0.9101, 6.7416)) == 2, '|-2.0899| and |-6.7416| exceed 1.0; |-0.9101| does not')
row('just under 1.4 times the reference', 'Sec. 4.6', 'w13-seeds-and-tables/TABLES.md', ['1.372', '1.358'],
    1.3 < 1.372 < 1.4, '1.358 to 1.372 times')
row('within half a percent (bytes and parameters)', 'Sec. 4.6', 'w13-seeds-and-tables/TABLES.md', ['715,808', '713,632', '719,352'],
    all(abs(v / 715808 - 1) <= 0.005 for v in (713632, 719352)) and all(abs(v / 178952 - 1) <= 0.005 for v in (178408, 179838)),
    'bytes 99.70% / 100.50%; parameters -0.304% / +0.495% (178,408 / 179,838 against 178,952)')
row('more than eighteen tuples (item memory, 3-seed mean)', 'Sec. 4.6', 'w13-seeds-and-tables/TABLES.md', ['18.56'],
    18 < 18.56, '3-seed mean 18.56')
row('far below one distinct valid novel tuple (LACE, PoE)', 'Sec. 4.7', 'w15-lace-poe-rows/FINDINGS.md', ['0.267'],
    max(0.267, 0.020) < 0.5, 'at most 0.267 and 0.020')
row('though its mean is positive (S2)', 'Sec. 4.7', 'w15-lace-poe-rows/FINDINGS.md', ['0.1551'], 0.1551 > 0, 'mean +0.1551')
row('about four times the reference bytes', 'Sec. 4.7', 'w15-lace-poe-rows/FINDINGS.md', ['4.09'],
    4 <= 4.09 < 4.5, '4.09--4.11 times')
row('about a third of the fit rows (seed-5 code)', 'Sec. 4.8', 'w23-seed5-diagnosis/FINDINGS.md', ['317', '211'],
    1 / 3 <= 106 / 317 < 0.35, '106 / 317 = 0.334 (106 = 317 - 211, as the existing SRC note states)')
row('small margin', 'Sec. 4.8', 'w23-seed5-diagnosis/FINDINGS.md', ['0.04'], True, 'margin 0.04')
row('small or absent on every seed but one; sizeable only on seed 2', 'Sec. 4.8', 'w21-rel-newseeds/FINDINGS.md', ['0.165'],
    sorted(abs(v) for v in (-0.007, 0.024, 0.165, 0.006, 0.0, 0.0))[-2] < 0.03, 'leads -0.007 / +0.024 / +0.165 / +0.006 / 0 / 0')
row('all or nearly all of the margin (seeds 3 and 4)', 'Sec. 4.8', 'w21-rel-newseeds/FINDINGS.md', ['0.97'],
    True, '0.97 and 1.00 (printed in the draft before, cited to w21-rel-newseeds/FINDINGS.md)')
row('well behind (value row, n-CLU-REL)', 'Sec. 4.8', 'w21-rel-newseeds/TABLES.md', ['0.554', '0.797'],
    0.797 - 0.554 > 0.2, '0.554 against 0.797')
row('no valid rollout; about nine in ten per step; a minority (n-CLU-REL)', 'Sec. 4.8', 'w21-maze-t-seed4/FINDINGS.md', ['0.900', '0.930'],
    0.9 <= 0.900 and max(0.067, 0.0, 0.289, 0.267, 0.156, 0.0) < 0.5, 'base 0.000; per step 0.900 to 0.930; REL at most 0.289')
row('under a second; a minute or more; thousands of gradient steps', 'Sec. 4.8', 'w21-maze-t-seed4/FINDINGS.md', ['0.88', '71'],
    0.88 < 1 and 71 >= 60, 'n-CLU 0.09 to 0.88 s; retrain 71 to 140 s and 8,000 steps')
row('less than one six-factor tuple (gentlest)', 'Sec. 5.3', 'w12-clue-theorist/CLUE.md', ['0.531'],
    max(0.531, 0.510, 0.388) < 1, '-0.531 / -0.510 / -0.388')
row('about forty on the seed mean (full strength)', 'Sec. 5.3', 'w12-clue-theorist/CLUE.md', ['39.959', '46.918', '39.061'],
    40 <= (39.959 + 46.918 + 39.061) / 3 < 45, 'mean 41.98; per seed 39.959 / 46.918 / 39.061')
row('well within its declared bar', 'Sec. 5.4', 'w10-physics-section-evidence/EVIDENCE.md', ['0.0433', '0.10'],
    max(0.0433, 0.0435, 0.0431) < 0.5 * 0.10, 'median log error 0.0433 / 0.0435 / 0.0431 against bar 0.10')
row('nearly two thirds of the image; most of it', 'Sec. 5.4', 'w10-physics-section-evidence/EVIDENCE.md', ['65.17', '65.39', '83.5'],
    0.6 <= 0.6539 < 2 / 3 and 0.835 > 0.5, '65.17% to 65.39%; 83.5%')

if '--plant' in sys.argv:  # positive control: a rounding that goes past the record must FAIL
    row('PLANTED: about seven (record 6.9)', '-', 'w20-shapes-data-matched/SHAPES-MATCHED.md', ['+7.034'], 7 <= 6.9, 'planted: 7 > 6.9')

fails = 0
print('| # | phrase in the text | where | record file | verbatim in record | rule holds | recorded values / rule |')
print('|---|---|---|---|---|---|---|')
for i, (ph, wh, f, nd, ok, why) in enumerate(R, 1):
    t = txt(f); found = [n for n in nd if n in t or n.replace(',', '') in t]
    v = 'yes' if found else 'NO'
    good = ok and bool(found)
    fails += not good
    print('| %d | %s | %s | `%s` | %s (%s) | %s | %s |' % (i, ph, wh, f, v, ', '.join(found) or '-', 'PASS' if good else 'FAIL', why))
print('\n%d rows, %d FAIL' % (len(R), fails))
sys.exit(1 if fails else 0)
