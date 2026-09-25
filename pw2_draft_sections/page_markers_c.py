"""PW-1 draft-final job 3, exact variant (C): page and position of every heading at SHIPOUT, without touching layout.

usage: python page_markers_c.py IN.tex OUT.tex > MARKS.json
Like page_markers.py, but the (B) marker goes INSIDE the heading's own text, where a whatsit adds no breakpoint:
  \\section{T}  -> \\section[T]{\\pwB{id}T}   (the optional argument keeps T out of the toc/bookmarks unchanged)
  \\section*{T} -> \\section*{\\pwB{id}T}
  bibliography  -> \\refname is prefixed with \\pwB{id} (natbib's \\bibsection prints \\section*{\\refname})
  end           -> \\pwB{id} on its own line before \\end{document} (can only affect the last page)
\\pwB is \\protected: \\pdfsavepos + a delayed \\write of \\thepage and \\pdflastypos (sp from the page bottom) to
\\jobname.pwm; the first line per id is the heading's. A 'D|' line records paperheight, voffset, topmargin,
headheight, headsep and textheight, so a position converts to a fraction of the text block. The (A) \\typeout lines are kept as well.
"""
import json, re, sys
src, dst = sys.argv[1:3]
text = open(src).read()
L = text.split('\n')
HEAD = re.compile(r'^(\s*)\\(section|subsection|subsubsection)(\*?)\{')
pre = (r'\makeatletter\newwrite\pw@marks\immediate\openout\pw@marks=\jobname.pwm'
       r'\protected\def\pwB#1{\pdfsavepos\write\pw@marks{B|#1|\thepage|\the\pdflastypos}}'
       r'\AtBeginDocument{\immediate\write\pw@marks{D|\the\paperheight|\the\voffset|\the\topmargin|\the\headheight|\the\headsep|\the\textheight}}\makeatother')
out, marks, i = [], [], 0
while i < len(L):
    l = L[i]; m = HEAD.match(l)
    if l.strip() == r'\begin{document}':
        out += [pre, l]; i += 1; continue
    if m:
        mid = 'm%03d' % len(marks)
        # gather lines until the heading argument's braces balance
        j = i; buf = l
        start = m.end() - 1
        def close(s):
            d = 0
            for k in range(start, len(s)):
                if s[k] == '{': d += 1
                elif s[k] == '}':
                    d -= 1
                    if d == 0: return k
            return None
        while close(buf) is None:
            j += 1; buf += '\n' + L[j]
        k = close(buf); arg = buf[start + 1:k]
        cmd = '\\' + m.group(2) + m.group(3)
        if m.group(3):
            new = m.group(1) + cmd + '{\\pwB{%s}' % mid + arg + '}' + buf[k + 1:]
        else:
            new = m.group(1) + cmd + '[' + arg + ']{\\pwB{%s}' % mid + arg + '}' + buf[k + 1:]
        marks.append({'id': mid, 'line': i + 1, 'kind': m.group(2) + m.group(3), 'title': ' '.join(arg.split())[:90]})
        out.append(r'\typeout{PW1MARK A|%s|\thepage}' % mid)
        out += new.split('\n'); i = j + 1; continue
    if l.strip().startswith(r'\IfFileExists{refs_clu_iclr.bib}'):
        mid = 'm%03d' % len(marks)
        marks.append({'id': mid, 'line': i + 1, 'kind': 'bibliography', 'title': 'References'})
        out.append(r'\typeout{PW1MARK A|%s|\thepage}' % mid)
        out.append(r'\let\pwrefname\refname\renewcommand{\refname}{\pwB{%s}\pwrefname}' % mid)
        out.append(l); i += 1; continue
    if l.strip() == r'\end{document}':
        mid = 'm%03d' % len(marks)
        marks.append({'id': mid, 'line': i + 1, 'kind': 'end', 'title': 'end of document'})
        out += [r'\typeout{PW1MARK A|%s|\thepage}' % mid, r'\pwB{%s}' % mid, l]; i += 1; continue
    out.append(l); i += 1
open(dst, 'w').write('\n'.join(out))
json.dump(marks, sys.stdout, indent=0)
