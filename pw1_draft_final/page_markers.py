"""PW-1 draft-final job 3: page counts per section, measured with markers on a SCRATCH copy.

usage: python page_markers.py IN.tex OUT.tex [--only-a]
  --only-a: typeout markers only (nothing enters any list; the layout is untouched). (B) whatsits can move a float:
  in DRAFT-v2 they moved app:encoder-main from p. 36 to p. 35, so the page counts are taken from an --only-a build.
Inserts, for every \\section / \\section* / \\subsection / \\subsubsection* heading, the bibliography and \\end{document}:
  (A) `\\typeout{PW1MARK A|<id>|\\thepage}` on its own line BEFORE the heading (immediate, adds nothing to any list);
  (B) `\\pwB{<id>}` on its own line AFTER the heading line: \\pdfsavepos + a delayed \\write of the shipped page and
      the y position (sp from the page bottom), so page and position are those of the shipped page.
Prints the marker table (id, line, kind, title) as JSON on stdout.
"""
import json, re, sys
src, dst = sys.argv[1:3]
ONLY_A = '--only-a' in sys.argv[3:]
L = open(src).read().split('\n')
HEAD = re.compile(r'^\s*\\(section|subsection|subsubsection)(\*?)\{(.*)')
out, marks = [], []
pre = (r'\makeatletter\newwrite\pw@marks\immediate\openout\pw@marks=\jobname.pwm'
       r'\newcommand{\pwB}[1]{\pdfsavepos\write\pw@marks{B|#1|\thepage|\the\pdflastypos}}\makeatother')
i = 0
while i < len(L):
    l = L[i]
    m = HEAD.match(l)
    kind = None
    if m:
        kind = m.group(1) + m.group(2); title = m.group(3)
    elif l.strip().startswith(r'\IfFileExists{references.bib}'):
        kind, title = 'bibliography', 'References'
    elif l.strip() == r'\end{document}':
        kind, title = 'end', 'end of document'
    elif l.strip() == r'\begin{document}':
        if not ONLY_A: out.append(pre)
        out.append(l); i += 1; continue
    if kind:
        mid = 'm%03d' % len(marks)
        marks.append({'id': mid, 'line': i + 1, 'kind': kind, 'title': title[:90]})
        out.append(r'\typeout{PW1MARK A|%s|\thepage}' % mid)
        if kind in ('bibliography', 'end'):
            if not ONLY_A: out.append(r'\pwB{%s}' % mid)
            out.append(l)
        else:
            # the heading may span lines: emit through the line where its braces close, then (B)
            buf = l; out.append(l)
            while buf.count('{') > buf.count('}') and i + 1 < len(L):
                i += 1; buf += L[i]; out.append(L[i])
            if not ONLY_A: out.append(r'\pwB{%s}' % mid)
        i += 1; continue
    out.append(l); i += 1
open(dst, 'w').write('\n'.join(out))
json.dump(marks, sys.stdout, indent=0)
