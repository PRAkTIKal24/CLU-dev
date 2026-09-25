#!/usr/bin/env python3
"""PW-1 item 7 completion: add the 0.04 margin and the 106-of-317 count to the seed-5 sentence.
usage: edit.py IN.tex OUT.tex   (the same one anchored operation for DRAFT-v2 and DRAFT-v2-style)"""
import sys
OLD = (
"and the miss is a content-capacity failure of the 6-atom design on that code; the 10-atom relational\n"
"design (n-CLU-REL, below) scores every row of both tuples valid. % pw1-draft-wording (item 7): mechanism as ruled in PW-1 PLAN item 7 (the W23 diagnosis); the code's 0.04 decoding margin and its 106 of 317 wrong fit rows are in no record on this machine and are not printed. Checked: ")
NEW = (
"and the miss is a content-capacity failure of the 6-atom design on that code: the code decodes at the wrong\n"
"side of a 0.04 margin in the deployed span and is already wrong on 106 of the 317 fit rows that carry it; % SRC: w23-seed5-diagnosis/FINDINGS.md; w23-seed5-diagnosis/Q1-CONTENT.json\n"
"% SRC note (pw1-draft-wording2, item 7): 106 is 317 - 211 (equally 63 + 43), arithmetic, not printed in the record. 0.04: w23-seed5-diagnosis/FINDINGS.md l.27 and l.70 (\"margin ... 0.04\"); JSON leaf w23-seed5-diagnosis/Q1-CONTENT.json codes[7] (code [1,3,0,2]) \"clu.pkl\".oracle_disp_margin = 0.0393, and W23-SEED5-DIAGNOSIS.json Q1.failing[\"2,0,1,2\"|\"2,0,3,2\"].l2_oracle_margins[1] = 0.0393. 317: Q1-CONTENT.json codes[7].n_train and W23-SEED5-DIAGNOSIS.json Q1.failing[...].truth_code_fit_rows. 211 / 63 / 43: Q1-CONTENT.json codes[7].clu_in_sample_emits_top3 (S = truth, E, N); FINDINGS.md l.30 \"211 / 317\" and l.71 \"317 rows; emits S 211, E 63, N 43\". Deployed span = the mean + 5 directions per well, deployed rank [5, 5] (FINDINGS.md l.25-26).\n"
"the 10-atom relational\n"
"design (n-CLU-REL, below) scores every row of both tuples valid. % pw1-draft-wording2 (item 7): the mechanism (content-capacity failure on one outcome code; not a hold-out effect) now rests on the W23 diagnosis record, w23-seed5-diagnosis/FINDINGS.md l.22-33 (\"CONTENT\"; \"not a hold-out effect at all\") and l.66-78, W23-SEED5-DIAGNOSIS.json Q1; it no longer rests on the PW-1 plan alone. Checked: ")
src, dst = sys.argv[1], sys.argv[2]
t = open(src).read()
n = t.count(OLD)
if n != 1:
    sys.exit('anchor count %d != 1 in %s' % (n, src))
open(dst, 'w').write(t.replace(OLD, NEW))
print('%s: 1 operation applied' % src)
