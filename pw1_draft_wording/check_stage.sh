#!/usr/bin/env bash
# usage: check_stage.sh SNAPDIR STAGEDIR   (STAGEDIR holds the edited drafts and build/ from compile_check.sh)
# Style diff-check, provenance before/after, captions before/after + banned-word scan, whole-draft vocabulary
# delta, PDF text ("Figure pending", placeholders), figure/section numbers and pages from the .aux.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; R="$(dirname "$HERE")"; SN="$1"; S="$2"; export PW1_SWEEPS="$R/pw1_draft_sweeps"
python3 "$R/pw1_draft_sweeps/style_diff_check.py" "$S/DRAFT-v2.tex" "$S/DRAFT-v2-style.tex" > "$S/STYLE-DIFF-CHECK.md"
echo "style diff-check: $(grep -c 'EMPTY DIFF' "$S/STYLE-DIFF-CHECK.md") EMPTY sections; non-True flags: $(grep -E '\*\*(False|NOT)' "$S/STYLE-DIFF-CHECK.md" | wc -l | tr -d ' ')"
for d in DRAFT-v2 DRAFT-v2-style; do
  python3 "$R/pw1_draft_sweeps/check_provenance.py" "$SN/$d.tex" > "$S/PROV-before-$d.txt"
  python3 "$R/pw1_draft_sweeps/check_provenance.py" "$S/$d.tex" > "$S/PROV-after-$d.txt"
  if diff <(sed 's/^[0-9]* //' "$S/PROV-before-$d.txt" | tail -n +2) <(sed 's/^[0-9]* //' "$S/PROV-after-$d.txt" | tail -n +2) > /dev/null
  then same=IDENTICAL; else same=DIFFERENT; fi
  echo "provenance $d: $(head -1 "$S/PROV-before-$d.txt") -> $(head -1 "$S/PROV-after-$d.txt"); findings $same; $(tail -1 "$S/PROV-after-$d.txt"); source-not-found $(grep -c 'SOURCE FILE NOT FOUND' "$S/PROV-after-$d.txt")"
  python3 "$R/pw1_draft_figures/captions.py" "$SN/$d.tex" "$S/CAPS-before-$d.json" "$S/CAPS-before-$d.tex" > /dev/null
  python3 "$R/pw1_draft_figures/captions.py" "$S/$d.tex" "$S/CAPS-after-$d.json" "$S/CAPS-after-$d.tex" > /dev/null
  python3 "$R/pw1_draft_sweeps/scan.py" "$S/CAPS-after-$d.tex" --lines > "$S/SCAN-captions-after-$d.txt"
  echo "banned-word hits in rendered captions, $d: $(awk '/^[^ ]/ && $NF>0' "$S/SCAN-captions-after-$d.txt" | grep -vE '^(Sec|Figs?|Tab|App|bare)' | awk '{s+=$NF} END{print s+0}')"
  paste <(python3 "$R/pw1_draft_sweeps/scan.py" "$SN/$d.tex") <(python3 "$R/pw1_draft_sweeps/scan.py" "$S/$d.tex" | awk '{print $NF}') \
    | awk -v d="$d" '{a=$(NF-1); b=$NF; if (a!=b) print d, $0}' > "$S/VOCAB-DELTA-$d.txt"
  uv run -q --with pypdf python "$R/pw1_draft_cuts/pdftext.py" "$S/build/$d.pdf" "$S/$d.pdf.txt" > /dev/null 2>&1
  echo "PDF $d: 'Figure pending' $(grep -c 'Figure pending' "$S/$d.pdf.txt"); pages $(grep -o 'Output written on .* (\([0-9]*\) pages' "$S/build/$d.log" | grep -o '([0-9]*' | tr -d '(')"
  grep -o 'newlabel{\(fig\|sec\|app\):[^}]*}{{[^}]*}{[0-9]*}' "$S/build/$d.aux" | sed 's/newlabel{//; s/}{{/ | /; s/}{/ | p./' > "$S/INDEX-$d.txt"
done
cmp -s "$S/CAPS-after-DRAFT-v2.json" "$S/CAPS-after-DRAFT-v2-style.json" && echo "captions identical in both drafts"
cmp -s "$S/INDEX-DRAFT-v2.txt" "$S/INDEX-DRAFT-v2-style.txt" && echo "fig/sec numbers and pages identical" || echo "fig/sec index differs between drafts (pages): see INDEX-*.txt"
for d in DRAFT-v2 DRAFT-v2-style; do
  python3 - "$S/$d.tex" "$S/PH-$d.txt" <<'EOF'
import re, sys
sys.path.insert(0, __import__('os').environ['PW1_SWEEPS'])
from texnum import split_comment
L = open(sys.argv[1]).read().split('\n')
code = '\n'.join(split_comment(l)[0] for l in L)
out = []
for kind in ('PLACEHOLDER', 'TODO'):
    for m in re.finditer(r'\\%s\{' % kind, code):
        if code[m.start()-12:m.start()].endswith('newcommand{'): continue
        i = m.end() - 1; dpt = 0
        for j in range(i, len(code)):
            if code[j] == '{': dpt += 1
            elif code[j] == '}':
                dpt -= 1
                if dpt == 0: break
        out.append((m.start(), kind, code.count('\n', 0, m.start()) + 1, ' '.join(code[i+1:j].split())))
out.sort()
open(sys.argv[2], 'w').write('\n'.join('%s L%d: %s' % (k, l, t) for _, k, l, t in out) + '\n')
print(sys.argv[1].split('/')[-1], 'PLACEHOLDER', sum(1 for o in out if o[1] == 'PLACEHOLDER'), 'TODO', sum(1 for o in out if o[1] == 'TODO'))
EOF
done
