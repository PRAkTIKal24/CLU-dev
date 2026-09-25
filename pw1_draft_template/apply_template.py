"""PW-1 draft-template: bring a draft's formatting to the ICLR 2027 template without changing a word.

usage: python apply_template.py IN.tex OUT.tex [--only header|lists|LINE[,LINE...]]
Two mechanical operations, each printed with its source line:
  header  load the template's OWN fancyhdr.sty and natbib.sty (iclr2027/) before iclr2027_conference.sty (same line).
          kpathsea finds the system fancyhdr (v5.2) before \\input@path is consulted; with v5.2 the sty's
          \\lhead{Under review ...}, set inside the \\@maketitle group, is local and is lost, so no page carries
          the running header. The template's bundled fancyhdr defines \\lhead globally, as the sample build shows.
  lists   drop the enumitem key list ([leftmargin=*,itemsep=..], [leftmargin=*,nosep]) from itemize/enumerate:
          the style file sets \\leftmargin, \\itemsep, \\topsep, \\parsep, which those keys override.
--only LINE,... removes only the list keys on those source lines (for per-override page-effect builds).
"""
import re
import sys

src, dst = sys.argv[1:3]
only = sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == '--only' else 'header,lists'
lines = open(src).read().split('\n')
out = []
HDR_OLD = r'\usepackage{iclr2027_conference,times}'
# on the SAME line as the class-style load: no line is added, so every provenance line number stays put
HDR_NEW = [r"\RequirePackage{iclr2027/fancyhdr,iclr2027/natbib}" + HDR_OLD +
           r" % fancyhdr and natbib: the template's own copies (its \lhead is global, so the running header prints)"]
LIST = re.compile(r'^(\s*\\begin\{(?:itemize|enumerate)\})(\[[^\]]*\])(.*)$')
want_lines = set()
do_header = 'header' in only
do_lists = 'lists' in only
if not (do_header or do_lists):
    want_lines = {int(x) for x in only.split(',')}
n_hdr = 0
for i, l in enumerate(lines, 1):
    if do_header and l.strip() == HDR_OLD:
        n_hdr += 1
        print(f'header  l.{i}: `{HDR_OLD}` -> `{HDR_NEW[0]}`')
        out.extend(HDR_NEW)
        continue
    m = LIST.match(l)
    if m and (do_lists or i in want_lines):
        print(f'list    l.{i}: removed `{m.group(2)}` from `{m.group(1).strip()}`')
        out.append(m.group(1) + m.group(3))
        continue
    out.append(l)
if do_header:
    assert n_hdr == 1, n_hdr
open(dst, 'w').write('\n'.join(out))
