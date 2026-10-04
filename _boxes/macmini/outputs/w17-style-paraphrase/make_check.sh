#!/bin/bash
# Rebuild DRAFT-v2-style.tex from the patches and regenerate STYLE-DIFF-CHECK.md (exit 1 on any non-empty diff).
set -e
cd "$(dirname "$0")"
python3 apply_patches.py ../w17-paper/DRAFT-v2.tex DRAFT-v2-style.tex
python3 style_diff_check.py ../w17-paper/DRAFT-v2.tex DRAFT-v2-style.tex > STYLE-DIFF-CHECK.md
cat style_notes.md >> STYLE-DIFF-CHECK.md
python3 check_provenance.py DRAFT-v2-style.tex > prov_style.txt
python3 check_provenance.py ../w17-paper/DRAFT-v2.tex > prov_orig.txt
diff <(sed -n '/UNSOURCED/,$p' prov_orig.txt | sed 's/^[0-9]* //') <(sed -n '/UNSOURCED/,$p' prov_style.txt | sed 's/^[0-9]* //') >/dev/null && echo "provenance: identical findings"
