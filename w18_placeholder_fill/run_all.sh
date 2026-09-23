#!/usr/bin/env bash
# w18 placeholder-fill: rebuild both drafts from the pre-edit snapshots, check, compile.
# usage: run_all.sh <build_dir>   (a fresh directory; nothing is deleted)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
O=/Users/batmac/clu/outputs/w18-placeholder-fill
P=/Users/batmac/clu/outputs/w17-paper
CHK=/Users/batmac/clu/outputs/w17-style-paraphrase
B="$1"; mkdir -p "$B"
python3 "$HERE/apply_fill.py" "$O/pre/DRAFT-v2.tex" "$B/DRAFT-v2.tex"
python3 "$HERE/apply_fill.py" "$O/pre/DRAFT-v2-style.tex" "$B/DRAFT-v2-style.tex"
( cd "$CHK" && python3 style_diff_check.py "$B/DRAFT-v2.tex" "$B/DRAFT-v2-style.tex" > "$B/STYLE-DIFF-CHECK.md" ) && echo "style check: ALL DIFFS EMPTY" || { echo "style check: NON-EMPTY"; exit 1; }
( cd "$CHK" && python3 check_provenance.py "$B/DRAFT-v2.tex" > "$B/prov_orig.txt" && python3 check_provenance.py "$B/DRAFT-v2-style.tex" > "$B/prov_style.txt" )
diff <(sed -n '/UNSOURCED/,$p' "$O/pre/prov_orig.txt" | sed 's/^[0-9]* //') <(sed -n '/UNSOURCED/,$p' "$B/prov_orig.txt" | sed 's/^[0-9]* //') && echo "provenance: findings identical to the pre-edit baseline"
diff <(sed -n '/UNSOURCED/,$p' "$B/prov_orig.txt" | sed 's/^[0-9]* //') <(sed -n '/UNSOURCED/,$p' "$B/prov_style.txt" | sed 's/^[0-9]* //') && echo "provenance: orig == style"
echo "PLACEHOLDER uses (macro_args): $(grep -c '^[0-9]*\. `w1' <(sed -n '/Placeholder keys/,/TODO/p' "$B/STYLE-DIFF-CHECK.md"))"
cp -R "$P/iclr2027" "$B/"; cp "$P/references.bib" "$B/"; ln -sfn "$P/figures" "$B/figures"
cp "$O/build/compile.sh" "$B/compile.sh"
( cd "$B" && ./compile.sh DRAFT-v2 && ./compile.sh DRAFT-v2-style && grep -c Overfull DRAFT-v2.log DRAFT-v2-style.log )
