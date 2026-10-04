#!/bin/bash
# stage1 -> pkgsrc (transform) -> nclu-iclr-code/ (package). Tooling only.
set -e
O=/Users/batmac/clu/outputs/w24-repro-package
A=$O/assembly
T=$O/tooling
P=$O/nclu-iclr-code
rm -rf $A/pkgsrc
python3 $T/transform.py $A/stage1 $A/pkgsrc > $A/transform_map.json
rm -rf $P
mkdir -p $P/weights
cp -R $A/pkgsrc/nclu $A/pkgsrc/scripts $A/pkgsrc/figures $P/
python3 $T/layout_weights.py $P/weights > /dev/null
python3 $T/sanitize_json_tree.py $P/weights
cp $T/static/pyproject.toml $T/static/LICENSE $P/
[ -f $T/static/README.md ] && cp $T/static/README.md $P/ || true
find $P -name '__pycache__' -prune -exec rm -rf {} \; 2>/dev/null || true
echo ASSEMBLED $(find $P -type f | wc -l) files
