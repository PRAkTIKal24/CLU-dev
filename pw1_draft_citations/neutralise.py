"""usage: neutralise.py IN.tex OUT.tex -- a control copy in which every natbib citation command and the
bibliography print NOTHING (definitions inserted just before \\begin{document}). Built before and after, the
rendered word diff of the two control copies can then show only text that is NOT printed by a citation command:
the hand-typed author/year strings the pass removes. Anything else would be a non-citation change."""
import sys
s = open(sys.argv[1]).read()
NEUT = ('\\makeatletter\\renewcommand\\citep[2][]{}\\renewcommand\\citet[2][]{}\\renewcommand\\citealp[2][]{}'
        '\\renewcommand\\cite[2][]{}\\renewcommand\\nocite[1]{}\\renewcommand\\bibliography[1]{}\\makeatother\n')
assert s.count('\n\\begin{document}') == 1
open(sys.argv[2], 'w').write(s.replace('\n\\begin{document}', '\n' + NEUT + '\\begin{document}'))
