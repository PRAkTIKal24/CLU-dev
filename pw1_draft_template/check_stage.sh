#!/usr/bin/env bash
# usage: bash check_stage.sh BEFORE_TREE AFTER_TREE OUTDIR
# Every acceptance check of pw1-draft-template, on two built w17-paper-layout trees (both drafts built in each):
#   compile metrics, overfull boxes, style diff-check, provenance (byte-compare of the checker's findings),
#   \PLACEHOLDER / \TODO arguments byte-identical, label numbers and pages, running header count,
#   the rendered word diff (worddiff.py), the formatting-level double-blind scan (blind_pdf.py), source diff.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; R="$(dirname "$HERE")"
B="$1"; A="$2"; O="$3"; mkdir -p "$O"
export PATH="/Library/TeX/texbin:/Users/batmac/clu/workspace/.venv/bin:$PATH"
{
echo "# stage checks: before=$B after=$A"
echo "## compile (after)"; cat "$A/COMPILE.txt"
echo "## compile (before)"; cat "$B/COMPILE.txt"
python3 "$R/pw1_draft_sweeps/style_diff_check.py" "$A/DRAFT-v2.tex" "$A/DRAFT-v2-style.tex" > "$O/STYLE-DIFF-CHECK-after.md"
echo "## style diff-check (after): $(grep -c 'result: \*\*EMPTY DIFF' "$O/STYLE-DIFF-CHECK-after.md") sections EMPTY; $(grep -o 'ALL DIFFS EMPTY' "$O/STYLE-DIFF-CHECK-after.md" || echo 'NOT ALL EMPTY')"
for d in DRAFT-v2 DRAFT-v2-style; do
  echo "## $d"
  grep '^Overfull' "$A/$d.log" > "$O/OVERFULL-$d.txt"; echo "overfull boxes (after): $(wc -l < "$O/OVERFULL-$d.txt" | tr -d ' ')"
  for t in B A; do T=${!t}
    python3 "$R/pw1_draft_sweeps/check_provenance.py" "$T/$d.tex" > "$O/PROV-$t-$d.txt"
    python3 "$R/pw1_draft_wording2/check_derived.py" "$T/$d.tex" "$O/PROV-$t-$d.txt" > "$O/PROVD-$t-$d.txt"
    python3 - "$T/$d.tex" > "$O/PHTODO-$t-$d.txt" <<'EOF'
import sys
s = open(sys.argv[1]).read()
for name in ('\\PLACEHOLDER{', '\\TODO{'):
    i = 0
    while (i := s.find(name, i)) >= 0:
        j, depth = i + len(name), 1
        while depth:
            depth += {'{': 1, '}': -1}.get(s[j], 0); j += 1
        if not s[:i].rsplit('\n', 1)[-1].lstrip().startswith('%') and 'newcommand' not in s[max(0, i - 12):i]:
            print(repr(s[i:j]))
        i = j
EOF
  done
  cmp -s "$O/PROV-B-$d.txt" "$O/PROV-A-$d.txt" && echo "provenance findings: BYTE-IDENTICAL ($(grep -E '^unsourced' "$O/PROV-A-$d.txt"))" || { echo "provenance findings DIFFER:"; diff "$O/PROV-B-$d.txt" "$O/PROV-A-$d.txt"; }
  cmp -s "$O/PROVD-B-$d.txt" "$O/PROVD-A-$d.txt" && echo "derived check: BYTE-IDENTICAL" || echo "derived check DIFFERS"
  cmp -s "$O/PHTODO-B-$d.txt" "$O/PHTODO-A-$d.txt" && echo "PLACEHOLDER/TODO arguments, in order: BYTE-IDENTICAL ($(grep -c PLACEHOLDER "$O/PHTODO-A-$d.txt") PLACEHOLDER, $(grep -c TODO "$O/PHTODO-A-$d.txt") TODO)" || echo "PLACEHOLDER/TODO DIFFER"
  python3 - "$B/$d.aux" "$A/$d.aux" <<'EOF'
import re, sys
L = re.compile(r'\\newlabel\{([^}]*)\}\{\{([^}]*)\}\{([^}]*)\}')
def lab(p):
    return {m.group(1): (m.group(2), m.group(3)) for m in map(L.match, open(p)) if m}
b, a = lab(sys.argv[1]), lab(sys.argv[2])
print(f'labels: {len(b)} -> {len(a)}; missing/new: {sorted(set(b) ^ set(a))}; renumbered: '
      f'{[k for k in b if k in a and b[k][0] != a[k][0]]}; page moved: {[(k, b[k][1], a[k][1]) for k in b if k in a and b[k][1] != a[k][1]]}')
EOF
  echo "running header lines: before $(grep -c 'Under review as a conference paper at ICLR 2027' "$B/$d.pdf.txt"), after $(grep -c 'Under review as a conference paper at ICLR 2027' "$A/$d.pdf.txt") (pages after: $(grep -c '^=== PAGE' "$A/$d.pdf.txt"))"
  python3 "$HERE/worddiff.py" "$B/$d.pdf.txt" "$A/$d.pdf.txt" > "$O/WORDDIFF-$d.txt"; echo "word diff exit $?: $(grep -E '^\(1\)|^RESULT|^moved' "$O/WORDDIFF-$d.txt" | tr '\n' ' ')"
  uv run -q --with pypdf python "$HERE/blind_pdf.py" "$A/$d.pdf" > "$O/BLIND-$d.txt" 2>&1; echo "blind scan: $(grep -E '/Info|token hits' "$O/BLIND-$d.txt" | cut -c1-260 | tr '\n' ' ')"
  diff "$B/$d.tex" "$A/$d.tex" > "$O/SOURCE-DIFF-$d.txt"; echo "source diff: $(grep -c '^<' "$O/SOURCE-DIFF-$d.txt") lines out, $(grep -c '^>' "$O/SOURCE-DIFF-$d.txt") lines in; before $(wc -l < "$B/$d.tex" | tr -d ' ') lines, after $(wc -l < "$A/$d.tex" | tr -d ' ')"
done
} | tee "$O/CHECKS.txt"
