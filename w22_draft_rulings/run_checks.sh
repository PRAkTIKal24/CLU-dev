#!/usr/bin/env bash
# usage: run_checks.sh <staged dir> : style diff, provenance (vs snapshot baseline), compile, retired-number grep
set -u; S="$1"; O=/Users/batmac/clu/outputs/w22-draft-rulings; H=$(cd "$(dirname "$0")" && pwd)
python3 $H/style_diff_check.py $S/DRAFT-v2.tex $S/DRAFT-v2-style.tex > $S/STYLE-DIFF-CHECK.md; echo "style_diff_exit=$?"
grep -E "placeholders;|TODO\{\}" $S/STYLE-DIFF-CHECK.md
for d in DRAFT-v2 DRAFT-v2-style; do
  python3 $H/check_provenance.py $S/$d.tex > $S/prov-$d.txt 2>&1
  if diff <(sed 's/^[0-9]* //' $O/prov-baseline-$d.txt | grep -v "checked number tokens") <(sed 's/^[0-9]* //' $S/prov-$d.txt | grep -v "checked number tokens") >/dev/null; then echo "prov $d: findings = baseline (line shifts only); $(head -1 $S/prov-$d.txt)"; else echo "prov $d: NEW FINDINGS"; diff <(sed 's/^[0-9]* //' $O/prov-baseline-$d.txt) <(sed 's/^[0-9]* //' $S/prov-$d.txt); fi
done
B=$S/build; rm -rf "$B" 2>/dev/null; bash $H/compile_check.sh $S $B 2>&1 | tail -2
for d in DRAFT-v2 DRAFT-v2-style; do echo "$d float-too-large/figure-pending: $(grep -c 'Float too large\|Figure pending' $B/$d.log)"; done
python3 $H/retired_grep.py $B/DRAFT-v2.pdf $B/DRAFT-v2-style.pdf > $S/RETIRED-GREP.txt; echo "retired_grep_exit=$? (hits listed in RETIRED-GREP.txt)"; grep -E "^==|^   line" $S/RETIRED-GREP.txt | grep -v ": 0 hit" 
