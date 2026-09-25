"""Whole-wave residual audit of rendered text (PW-1 items 2, 3, 5, 8, 13, 14, 15, 17 and the resolved pendings).

usage: python residual_audit.py DRAFT.tex DRAFT.pdf.txt OUT.json
Two channels per draft:
  src -- the .tex with comments stripped (texnum.split_comment); \\label/\\ref/\\cite keys blanked EXCEPT for the
         'removed key' rules, which read the raw code so a dangling key counts;
  pdf -- the PDF text (pypdf; figure text of vector figures included), whitespace-normalised, and a second copy with
         line-break hyphens joined ('com-\\nbination' -> 'combination'); a hit counts once per channel copy, the
         reported count is the larger of the two copies.
Every hit is written with 70 characters of context. Known blind spots of the pdf channel (shown by the positive
control): a word hyphenated across a page break with a float between the halves; '\\_' (pypdf extracts the drawn
underscore as a space). The src channel covers both. No rule is a pass/fail label: counts and contexts only.
"""
import json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pw1_draft_sweeps'))
from texnum import split_comment  # noqa: E402

I = re.I
# (rule, item, src regex or None, pdf regex or None, flags)
R = [
 ('rival', '2', r'\brivals?\b', r'\brivals?\b', I),
 ('store', '2', r'\bstor(e|es|ed|ing)\b', r'\bstor(e|es|ed|ing)\b', I),
 ('gate', '2', r'\bgat(e|es|ed|ing)\b', r'\bgat(e|es|ed|ing)\b', I),
 ('artifact/artefact', '2', r'\bartef?i?facts?\b|\bartefacts?\b', r'\bartef?i?facts?\b|\bartefacts?\b', I),
 ('pre-registered', '2', r'\bpre-?regist\w*', r'\bpre-?\s?regist\w*', I),
 ('A1', '2', r'\bA1\b', r'\bA1\b', 0), ('A10', '2', r'\bA10\b', r'\bA10\b', 0),
 ('C-512', '2', r'\bC-512\b', r'\bC-512\b', 0), ('improved', '2', r'\bimproved\b', r'\bimproved\b', I),
 ('config (abbrev.)', '2', r'\bconfigs?\b', r'\bconfigs?\b', I),
 ('configuration (English word)', '2', r'\bconfigurations?\b', r'\bconfigurations?\b', I),
 ('spoke', '2', r'\bspokes?\b', r'\bspokes?\b', I), ('hub', '2', r'\bhubs?\b', r'\bhubs?\b', I),
 ('wave', '2', r'\bwaves?\b', r'\bwaves?\b', I), ('charter', '2', r'\bcharter\w*', r'\bcharter\w*', I),
 ('ledger', '3', r'\bledgers?\b', r'\bledgers?\b', I), ('bar k', '2', r'\bbar [a-zA-Z]\b|\bbar-[a-z]\b', r'\bbar [a-zA-Z]\b|\bbar-[a-z]\b', 0),
 ('scored run', '2', r'\bscored runs?\b', r'\bscored runs?\b', I), ('budget', '2', r'\bbudget\w*', r'\bbudget\w*', I),
 ('spend/spent', '2', r'\bsp(end|ent|ends|ending)\b', r'\bsp(end|ent|ends|ending)\b', I),
 ('price/priced/pricing', '2', r'\bpric(e|es|ed|ing)\b', r'\bpric(e|es|ed|ing)\b', I),
 ('buy/buys/bought', '2', r'\bbu(y|ys|ying|ought)\b', r'\bbu(y|ys|ying|ought)\b', I),
 ('pay/paid', '2', r'\bpa(y|ys|id|ying)\b', r'\bpa(y|ys|id|ying)\b', I),
 ('cost (technical only)', '2', r'\bcost\w*', r'\bcost\w*', I), ('cheap', '2', r'\bcheap\w*', r'\bcheap\w*', I),
 ('expensive', '2', r'\bexpensive\b', r'\bexpensive\b', I),
 ('t_pay / price law / priced channel', '2', r't_\{?\\(mathrm|text)\{pay\}|price law|priced channel', r'\bt\s?pay\b|price law|priced channel', I),
 ('corner', '3', r'\bcorners?\b', r'\bcorners?\b', I), ('held-out composition', '3', r'held-out compositions?', r'held-out compositions?', I),
 ('oblong', '3', r'\boblongs?\b', r'\boblongs?\b', I),
 ('exact judge', 'pending', r'\bexact(ly)? judge', r'\bexact(ly)? judge', I),
 ('exact / exactly / exactness (all senses)', '13', r'\bexact(ly|ness)?\b', r'\bexact(ly|ness)?\b', I),
 ('10x / 10x steps', '2', r'\b10x\b|10\s*\$?\\times\$?\s*steps\b', r'\b10x\b|10\s?×\s?steps\b', I),
 ('internal control names', '2', r'landscape\\?_(removed|shuffled)|random\\?_atoms|shuffled\\?_across', r'landscape_(removed|shuffled)|random_atoms|shuffled_across', 0),
 ('Naive CLU / naive CLU', '2', r'\bnaive CLU\b', r'\bnaive CLU\b', I),
 ('bare CLU', '2', r'(?<![-\w])CLU(?![-\w])', r'(?<![-\w])CLU(?![-\w])', 0),
 ('hard (any sense)', '8', r'\bhard\w*', r'\bhard\w*', I), ('easy (any sense)', '8', r'\beas(y|ier|iest)\b', r'\beas(y|ier|iest)\b', I),
 ('difficult*', '8', r'\bdifficult\w*', r'\bdifficult\w*', I),
 ('ceiling', '1.5/8', r'\bceilings?\b', r'\bceilings?\b', I), ('best fitted arm', '1.5', r'best fitted arm', r'best fitted arm', I),
 ('WIN / LOSE / COMPARABLE / LOSS labels', '1.3', r'\\textsc\{(win|lose|loss|comparable)\}|\b(WIN|LOSE|COMPARABLE|LOSS)\b', r'\b(WIN|LOSE|COMPARABLE|LOSS)\b', 0),
 ('CMAZE-1 / 2 / 2b / O', '14', r'CMAZE-(1|2b?|O)\b', r'CMAZE-(1|2b?|O)\b', 0),
 ('route', '14', r'\broutes?\b', r'\broutes?\b', I), ('verb', '14', r'\bverbs?\b', r'\bverbs?\b', I),
 ('request layer', '14', r'request[- ]layer', r'request[- ]layer', I),
 ('red ∧ sphere', '12.5', r'red\s*\$\\wedge\$\s*sphere', r'red\s?∧\s?sphere', I),
 ('Exactly Deletable (title)', '13', r'Exactly Deletable', r'Exactly Deletable', I),
 ('§ / \\S', '5', r'\\S(?![a-zA-Z])', r'§', 0),
 ('Section~ / Section N', '5', r'Sections?~\\ref', r'\bSections? \d', 0),
 ('Figure~ / Figure N (not a caption label)', '5', r'Figures?~\\ref', r'\bFigures? \d+(?![\d:])', 0),
 ('Table~ / Table N (not a caption label)', '5', r'Tables?~\\ref', r'\bTables? \d+(?![\d:])', 0),
 ('Appendix~ / Appendix X', '5', r'Appendi(x|ces)~\\ref', r'\bAppendi(x|ces) [A-Z]\b', 0),
 ('ref without ~ after Sec./Fig./Tab./App. (breaking space)', '5', r'\b(Secs?|Figs?|Tabs?|Apps?|Eqs?|Algs?)\. \\ref', None, 0),
]
KEYS = [('removed keys/files (items 14, 15, 17)', '14/15/17',
         r'maze-rel-tuples|cmazet-deletion|fig_maze_rel_tuples|fig_cmazet_deletion|readverb|requestlayer|\{tab:maze-del\}|\{tab:verb\}|\{tab:cmazeo\}|tab:maze-del-2b|tab:req2b|\{tab:req\}|\{fig:verb\}|impl-maze')]

def ctx(s, a, b, w=70):
    return ' '.join(s[max(0, a - w):min(len(s), b + w)].split())

def run(tex, pdftxt):
    L = open(tex).read().split('\n')
    raw = '\n'.join(split_comment(l)[0] for l in L)
    code = re.sub(r'\\(label|ref|eqref|pageref|citep|citet|cite|nocite)\{[^}]*\}', lambda m: '\\' + m.group(1) + '{}', raw)
    p0 = open(pdftxt).read()
    pa = ' '.join(p0.split())
    pb = ' '.join(re.sub(r'(?<=[a-z])-\s*\n\s*(?=[a-z])', '', p0).split())
    out = []
    for name, item, rs, rp, fl in R:
        rec = dict(rule=name, item=item, src=[], pdf=[])
        if rs:
            # a multi-word token may be split across source lines: a literal space matches any whitespace run
            for m in re.finditer(rs.replace(' ', r'\s+'), code, fl):
                rec['src'].append(dict(line=code.count('\n', 0, m.start()) + 1, hit=m.group(0), ctx=ctx(code, m.start(), m.end())))
        if rp:
            ha = [dict(hit=m.group(0), ctx=ctx(pa, m.start(), m.end())) for m in re.finditer(rp, pa, fl)]
            hb = [dict(hit=m.group(0), ctx=ctx(pb, m.start(), m.end())) for m in re.finditer(rp, pb, fl)]
            rec['pdf'] = hb if len(hb) > len(ha) else ha
        out.append(rec)
    for name, item, rx in KEYS:
        rec = dict(rule=name, item=item, src=[], pdf=[])
        for m in re.finditer(rx, raw):
            rec['src'].append(dict(line=raw.count('\n', 0, m.start()) + 1, hit=m.group(0), ctx=ctx(raw, m.start(), m.end())))
        out.append(rec)
    checks = dict(
        title_source=' '.join(re.sub(r'\\\\', ' ', re.search(r'\\title\{(.*?)\}\n', raw, re.S).group(1)).split()),
        title_is_the_recorded_one=' '.join(re.sub(r'\\\\', ' ', re.search(r'\\title\{(.*?)\}\n', raw, re.S).group(1)).split())
            == 'naive-CLU: Deletable Compositional Generation on a Frozen Encoder',
        first_mention_naive_CLU_n_CLU=bool(re.search(r'naive-CLU\s*\(n-CLU\)', pa)),
        engines_judge_count=len(re.findall(r"engine[’']s judge", pa)),
        ten_x_training_steps_count=len(re.findall(r'10\s?×\s?training steps', pa)),
    )
    return out, checks

if __name__ == '__main__':
    tex, pdftxt, oj = sys.argv[1:4]
    out, checks = run(tex, pdftxt)
    json.dump(dict(rules=out, checks=checks), open(oj, 'w'), indent=1, ensure_ascii=False)
    for r in out: print('%-58s src %4d  pdf %4d' % (r['rule'], len(r['src']), len(r['pdf'])))
    print(checks)
