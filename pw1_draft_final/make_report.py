"""Writes FINAL-INDEX.md (jobs 1-6 as tables) from a run_final.sh directory. usage: python make_report.py FINALDIR OUT.md
The residual REASONS below were written after reading every hit in AUDIT-*.json; a rule with a hit and no reason
prints 'UNCLASSIFIED' (none may remain)."""
import json, os, re, sys
F, OUTMD = sys.argv[1:3]
D = ('DRAFT-v2', 'DRAFT-v2-style')
L = []
def w(s=''): L.append(s)
def rd(p): return open(os.path.join(F, p)).read()

REASONS = {
 'config (abbrev.)': 'PDF only: pypdf splits the English word at a line-break hyphen ("config- urations"); the source has 0 abbreviations.',
 'configuration (English word)': 'The English word ("configurations", table header "configurations"); the plan bans the abbreviation "config", which is 0. Kept as in pw1-draft-sweeps.',
 'spend/spent': 'PDF only, inside the Fig. 2 image (fig_task_cmazet): the engine\'s event names "key spent at door" and "event spend". Item 6 rules Fig. 2 final with no changes.',
 'cost (technical only)': 'Technical sense in all 7 source hits: seconds per seed (Tab. del "cost" row block, Tab. 13 "Deletion cost ... retrain"), bytes ("two holes cost 2 bytes", "the coupling hole costs one byte", "cost line"), "memory cost", "memory and time cost". The 2 extra PDF hits are the author name "Costa" in the bibliography.',
 'cheap': '"Deletion is not cheap in memory or time" (limitation 3): "not cheap" is allowed by item 2.',
 'corner': 'PDF only, inside the Fig. 1 image (fig_task_shapes): "shape in the corner", "pair corner (2-way)", "six-factor corner (n-way)". Item 3: "Fig. 1 itself: no changes." The source has 0.',
 'exact / exactly / exactness (all senses)': 'No deletion sense (34 = 9 + 4 + 5 + 16): numeric equality (9: "exactly 0.0000" / "exactly 0.000" x3, "exactly constant" x2, "exactly 16,384 B", "exactly 0.0" x2, "exactly zero off the coupling graph"), the same rows (4: "exact 131,072-row pool", "exactly the n-CLU\'s content rows" x3), exact-match requests (5), mathematics/physics (16: zero-dynamics limit, returns z_req exactly at lambda=0, exactly delta/(1+beta), sqrt(1-gamma), exact log-derivative, exact inversion, exact root, bit-exactly (replication), ties exactly on two refusal rows, exact one-step Jacobian, exact only at zero damping, exact value fidelity (trilemma), 39% exactly optimal, exact-key lookup, exact quadratic, "were then exact"); the 34 of pw1-draft-wording\'s list. PDF: 33 of the 34 (pypdf drops the space in "exactly delta") + 2 cited titles in the bibliography ("Exact federated continual unlearning ...", "... Efficient and exact machine unlearning ...").',
 '10x / 10x steps': 'PDF only, fig_ladder legend "seed at 10x steps": item 10.6 prescribes this entry verbatim ("seed at $10\\times$ steps"). Item 1.8 says never "10x steps" for the maze figure. FOR THE HEAD: the two rulings differ; the figure follows item 10.6.',
 'bare CLU': 'Source (3): the CLU class / the settling physics CLU (Sec. 5.2 "limit of the CLU", Sec. 6 title and first sentence), not the naive unit. PDF extra hits are pypdf splitting "n-CLU" at a line-break hyphen ("n- CLU").',
 'hard (any sense)': '"Hard read isolation (designed memory only)", App. L (physics trilemma), a memory property; no seed is called hard.',
 'ceiling': 'No per-seed / best-of-arms maze ceiling (item 1.5): Shapes3D count ceilings 36 / 72 (Sec. 2, Tab. 1 caption, App. F header x2, App. I caption), "its 24-tuple ceiling" (Sec. 3.2), "a ceiling and not an arm" (item memory, Sec. 3.6), nearest-prototype ceiling (Sec. 3.7, App. B caption and Tab. lace-poe x2), "the ceiling ladder" (App. F title), "oracle read (ceiling)", "absolute content ceiling at the pin", "Two ceiling rungs" (App. F), and Tab. 19 (w16 asymmetric row) "Ceiling set by the request design, not the n-CLU" (a design plateau, flagged by pw1-draft-wording for PW-2).',
 'WIN / LOSE / COMPARABLE / LOSS labels': 'FOR THE HEAD: declared-criterion outcome words in small caps, not item 1.3\'s figure label column: Sec. 3.7 "(-0.1150, S3 LOSE)"; App. B, Tab. lace-poe caption "S3 ... LOSE; S4 ... WIN" and its last row "LOSE" (with PASS/FAIL beside them). Outside every PW-1 item; not a mechanical defect, so not edited.',
 'route': 'Six in the English sense of a deletion path (Sec. 1.1 x2, Sec. 3.4 x4: "only route to that match is the retrain", "Our route matches the refit", ...); one in the mandated CMAZE-suite paragraph ("route planning", item 14 wording). No route-task result remains.',
 '§ / \\S': 'Inside the abstract\'s CMAZE-T \\TODO (Head-held, byte-identical by ruling): "(\\S\\ref{sec:maze-wm}, Table~\\ref{tab:cmazet-sym})".',
 'Table~ / Table N (not a caption label)': 'The same abstract \\TODO as above ("Table 20" in the PDF). Every other "Table N" in the PDF is LaTeX\'s own caption label "Table N:".',
}
# ---------- header
w('# FINAL-INDEX — PW-1 closing audit (pw1-draft-final)')
w()
w('Session `pw1-draft-final` · batmac · branch `pw1/draft-final` · the drafts as installed after this pass (the title line-break fix, §1.3). '
  'Everything below is read from the compiled `.aux`/`.log` of the live tree and from `%s/`.' % os.path.basename(F.rstrip('/')))
w('Title (recorded for the report): **naive-CLU: Deletable Compositional Generation on a Frozen Encoder**.')
w()
# ---------- 1
w('## 1. Final compile (both drafts, scratch copy and in place)')
w()
w('| build | draft | exits (pdflatex 1, bibtex, later pdflatex) | pdflatex passes | undefined citations | undefined references | errors | overfull boxes | underfull (info) | bib items | pages |')
w('|---|---|---|---|---|---|---|---|---|---|---|')
for tag, f in (('scratch (clean copy of the w17-paper tree)', 'COMPILE-scratch.txt'), ('in place (`~/clu/outputs/w17-paper/`)', 'COMPILE-inplace.txt')):
    for l in rd(f).strip().split('\n'):
        d = l.split(' | ')[0]; kv = dict(x.split('=') for x in l.split(' | ')[1].split())
        ex = re.search(r'pdflatex1=(\d+) bibtex=(\d+) later_pdflatex=\[([^\]]*)\] passes=(\d+)', l)
        w('| %s | %s | %s, %s, %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (tag, d, ex.group(1), ex.group(2), ex.group(3).strip(), ex.group(4),
          kv['ucit'], kv['uref'], kv['errors'], kv['overfull'], kv['underfull'], kv['bibitems'], kv['pages']))
w()
cmpl = rd('JOB1-compare.txt').strip().split('\n')
w('- Scratch vs in place: ' + '; '.join(cmpl) + '. **Page counts equal (42 = 42 in both drafts).** The PDF bytes differ only through the embedded date/ID (text, `.aux`, `.bbl`, `.out` byte-equal).')
w('- Overfull boxes: **none** in either draft (both builds), so there is nothing to list.')
sd = rd('STYLE-DIFF-CHECK.md')
w('- Style diff-check (`pw1_draft_sweeps/style_diff_check.py`, DRAFT-v2 vs DRAFT-v2-style): **%s** (numbers, placeholders, TODOs, artefact paths, provenance comments in order, headings, labels, floats, preamble).'
  % ('ALL DIFFS EMPTY' if 'ALL DIFFS EMPTY' in sd else 'NOT EMPTY'))
for d in D:
    pr = rd('PROV-%s.txt' % d).split('\n'); dv = rd('PROV-DERIVED-%s.txt' % d).split('\n')
    w('- Provenance %s: %s; %s; %s. %s (identical to the snapshot before this pass).' % (d, pr[0], [x for x in pr if x.startswith('unsourced')][0] if any(x.startswith('unsourced') for x in pr) else pr[-2], dv[0], dv[1] if len(dv) > 1 else ''))
w('- The one derived number (Hub note): **106 = 317 − 211**, raw provenance "not found" +1, verified by `pw1_draft_wording2/check_derived.py` (both operands found in the cited W23 files). Reported, not changed.')
w()
w('### 1.3 The one draft edit of this pass (mechanical defect fix)')
w('The title rendered on three lines, `NAIVE-CLU: DELETABLE COMPOSITIONAL GENERA-` / `TION` / `ON A FROZEN ENCODER`: item 13 dropped "Exactly" and kept the forced break after "Generation"; the small-caps first line then overfills and LaTeX hyphenates. '
  'Fix (no word changed, identical in both drafts): `\\title{naive-CLU: Deletable Compositional Generation\\\\ on a Frozen Encoder}` → `\\title{naive-CLU: Deletable Compositional\\\\ Generation on a Frozen Encoder}` (DRAFT-v2 l.75, DRAFT-v2-style l.81). '
  'Staged, built from the w17-paper layout, checked (0/0 undefined, 0 overfull, 42 pp, style diff ALL EMPTY, provenance identical, `\\PLACEHOLDER`/`\\TODO` 2/4 unchanged, **0 labels moved**; the PDF text differs only in the two title lines), installed byte-identically into both drafts. Images: `title-DRAFT-v2.png` (before), `stage-title-*/title-after-*.png` (after).')
w()
# ---------- 2
ix = json.load(open(os.path.join(F, 'FIGTAB-INDEX.json')))
a, b = ix['DRAFT-v2']['items'], ix['DRAFT-v2-style']['items']
w('## 2. Figure and table index (from the compiled `.aux`; every row cross-checked against the list of figures/tables: 0 disagreements)')
w()
for kind, title in (('figure', 'Figures'), ('table', 'Tables')):
    w('### %s' % title); w()
    w('| DRAFT-v2 no. | label | caption, first line (bold lead) | page | section where the float is in the source | style no. | style page |')
    w('|---|---|---|---|---|---|---|')
    rows = sorted([(int(v['num']), k, v) for k, v in a.items() if v['kind'] == kind])
    for n, k, v in rows:
        s = b.get(k, {})
        sec = '%s %s' % (v['section'], re.sub(r'\s+', ' ', v['section_title'])[:48])
        w('| %s | `%s` | %s | %d | %s | %s | %s |' % (v['num'], k, v['lead'].replace('|', '/'), v['page'], sec.replace('|', '/'), s.get('num'), s.get('page')))
    w()
w('`tab:cmaze-del` is a second `\\label` on Tab. 19 (`tab:cmazet`, DRAFT-v2 l.2162); no `\\ref` uses it. Figure and table numbers are identical in both drafts; pages differ by at most one (the style draft\'s Sec. 3 sets differently).')
w()
# ---------- 3
w('## 3. Page counts per section (markers on a scratch copy)')
w()
w('Method: before every heading `\\typeout{PW1MARK A|id|\\thepage}` (A, as briefed) and, inside the heading text, `\\pdfsavepos` + a delayed `\\write` of the SHIPPED page and the y position (C). '
  'The marker builds move no label (every `\\newlabel` equal to the unmarked build) and C equals the `.aux` toc page at every numbered heading (0 mismatches). '
  'A lags C at 17 of 47 markers (DRAFT-v2) and 14 of 47 (style), because `\\thepage` is read before the page builder ships the page; **C is the page count used**. '
  'Length = difference of text-block coordinates, (page − 1) + (y_top − y)/\\textheight, to the next heading of the same or higher level, in pages; it includes any floats that TeX placed in between.')
w()
P = {d: json.load(open(os.path.join(F, 'PAGES-%s.json' % d))) for d in D}
w('| section | title | DRAFT-v2 start page (C) | `\\typeout` page (A) | length, pages | style start (C) | style A | style length |')
w('|---|---|---|---|---|---|---|---|')
for r, s in zip(P['DRAFT-v2']['rows'], P['DRAFT-v2-style']['rows']):
    assert r['title'] == s['title']
    t = re.sub(r'\s+', ' ', r['title']).replace('|', '/')[:70]
    ind = '&nbsp;&nbsp;' if r['kind'].startswith('subsection') else ''
    lab = r['num'] or {'subsubsection*': '(stmt)', 'bibliography': '(refs)', 'section*': '', 'subsection*': '', 'end': '(end)'}.get(r['kind'], '')
    f2 = lambda x: '' if x is None else '%.2f' % x
    w('| %s%s | %s | %d | %s | %s | %d | %s | %s |' % (ind, lab, t, r['page'], r['A_page'], f2(r['length']), s['page'], s['A_page'], f2(s['length'])))
w()
def find(d, t): return next(r for r in P[d]['rows'] if r['title'].startswith(t))
for d in D:
    ai = find(d, 'AI use statement'); bib = find(d, 'References'); app = find(d, 'Appendices'); end = [r for r in P[d]['rows'] if r['kind'] == 'end'][0]
    w('- **%s:** main text (title to the first statement heading, "AI use statement") = **%.2f pages**, ending on p. %d; ICLR 2027 initial submission limit 9 pages (VENUE.md §1) → **%.2f pages over** (the main text is %.0f%% of the limit). '
      'Statements (AI use, ethics, reproducibility; not counted by ICLR): %.2f pp. References: pp. %d–%d. Appendices: pp. %d–%d (%d pages). **Total pages: %d.**'
      % (d, ai['coord'], ai['page'], ai['coord'] - 9, 100 * ai['coord'] / 9, bib['coord'] - ai['coord'], bib['page'], app['page'] - 1, app['page'], end['page'], end['page'] - app['page'] + 1, P[d]['pages_total']))
w()
# ---------- 4
w('## 4. Every remaining `\\TODO` and `\\PLACEHOLDER` (counted by argument, brace-balanced, comments stripped)')
w()
w('Positive control: two `\\TODO{}`s planted on one line and one `\\PLACEHOLDER{}` → arguments TODO 4 → 6, PLACEHOLDER 2 → 3, while line counts go 4 → 5 (`grep -c` would miss one).')
w()
w('| # | kind | DRAFT-v2 line | style line | section | in float | argument (full) |')
w('|---|---|---|---|---|---|---|')
pa = json.load(open(os.path.join(F, 'PH-DRAFT-v2.json'))); pb = json.load(open(os.path.join(F, 'PH-DRAFT-v2-style.json')))
assert [x['arg'] for x in pa['items']] == [x['arg'] for x in pb['items']]
for i, (x, y) in enumerate(zip(pa['items'], pb['items']), 1):
    w('| %d | %s | %d | %d | %s | %s | %s |' % (i, x['kind'], x['line'], y['line'], x['section'].replace('|', '/'), x['in_float'] or '—', x['arg'].replace('|', '/')))
w()
w('Counts: DRAFT-v2 %s; DRAFT-v2-style %s (arguments byte-identical and in the same order). Line counts: %s / %s.' % (pa['count_args'], pb['count_args'], pa['count_lines'], pb['count_lines']))
w()
# ---------- 5
w('## 5. Whole-wave residual audit of rendered text (items 2, 3, 5, 8, 13, 14, 15, 17 and the pendings)')
w()
w('Channels: **src** = the `.tex` without comments (`\\label`/`\\ref` keys blanked, except the removed-keys rule, which reads them); **pdf** = the in-place PDF text (pypdf, vector-figure text included). '
  'Positive control: one paragraph planting every token, compiled into scratch copies of both drafts: **every rule caught its token in src; in pdf every rule but two** — `\\_` names (pypdf extracts the drawn underscore as a space, so `landscape_removed` reads as the allowed plain name) and a word hyphenated across a page break with a float between the halves ("expen-/sive", DRAFT-v2 only). The src channel covers both.')
w()
A = {d: json.load(open(os.path.join(F, 'AUDIT-%s.json' % d))) for d in D}
w('| rule | item | DRAFT-v2 src | DRAFT-v2 pdf | style src | style pdf | residuals: every hit, with its reason |')
w('|---|---|---|---|---|---|---|')
uncl = []
for r, s in zip(A['DRAFT-v2']['rules'], A['DRAFT-v2-style']['rules']):
    n = len(r['src']) + len(r['pdf']) + len(s['src']) + len(s['pdf'])
    reason = REASONS.get(r['rule'], 'UNCLASSIFIED' if n else '0')
    if n and r['rule'] not in REASONS: uncl.append(r['rule'])
    w('| %s | %s | %d | %d | %d | %d | %s |' % (r['rule'].replace('|', '/'), r['item'], len(r['src']), len(r['pdf']), len(s['src']), len(s['pdf']), reason.replace('|', '/')))
w()
c = A['DRAFT-v2']['checks']; c2 = A['DRAFT-v2-style']['checks']
w('Presence checks: title (source) = "%s" (both drafts: %s / %s); first mention "naive-CLU (n-CLU)" present: %s / %s; "the engine\'s judge" %d / %d times; "$10\\times$ training steps" %d / %d times.'
  % (c['title_source'], c['title_is_the_recorded_one'], c2['title_is_the_recorded_one'], c['first_mention_naive_CLU_n_CLU'], c2['first_mention_naive_CLU_n_CLU'],
     c['engines_judge_count'], c2['engines_judge_count'], c['ten_x_training_steps_count'], c2['ten_x_training_steps_count']))
w('Every hit with 70 characters of context: `AUDIT-DRAFT-v2.json`, `AUDIT-DRAFT-v2-style.json`. Unclassified residuals: **%s**.' % (', '.join(uncl) or 'none'))
w()
# ---------- 6
w('## 6. Figure-folder audit (`w17-paper/figures/` and `paper-visuals/`)')
w()
fo = json.load(open(os.path.join(F, 'FOLDERS.json')))
w('| placed file | sha256 (16 hex), w17-paper/figures | paper-visuals | equal |')
w('|---|---|---|---|')
for r in fo: w('| `%s` | %s | %s | %s |' % (r['file'], (r['w17'] or 'MISSING')[:16], (r['pv'] or 'MISSING')[:16], 'yes' if r['equal'] else '**NO**'))
w()
for l in rd('JOB6-folders.txt').strip().split('\n'): w('- ' + l)
w('- Positive control: in a copy of paper-visuals, one byte appended to `fig_ladder.png` and a line appended to README.md → the file is flagged unequal, both manifests flag it, and the README check reads False.')
w('- Step 0 (Hub note): `bash build.sh` in place, default destinations: exit 0, both builds byte-identical; **26 / 26 placed files byte-unchanged in each folder**; the new `PROVENANCE.json` (`ec5a6122…`) is cmp-equal to the one pw1-visuals-tooling predicted, and its JSON diff against the old one is exactly the three edits; only `PROVENANCE.json`, `SHA256-w22-build{1,2}.txt` and (after make_index) `README.md` changed in each folder.')
open(OUTMD, 'w').write('\n'.join(L) + '\n')
print('written', OUTMD, 'unclassified:', uncl)
