#!/bin/bash
# stage0 -> stage0p (patched) -> stage1 (shaken). Tooling only.
set -e
A=/Users/batmac/clu/outputs/w24-repro-package/assembly
T=/Users/batmac/clu/outputs/w24-repro-package/tooling
rm -rf $A/stage0p $A/stage1
cp -R $A/stage0 $A/stage0p
python3 $T/patches.py $A/stage0p
python3 $T/shake.py $A/stage0p $A/stage1 $A/entries_all.txt --keep-all chlu.data.shapes3d > $A/shake_report.txt
cp $A/stage0p/chlu/data_manifest.json $A/stage1/chlu/ 2>/dev/null || true
cp $A/stage0p/scripts/CLU-REL-A10.json $A/stage1/scripts/ 2>/dev/null || true
mkdir -p $A/stage1/chlu/specs && cp $A/stage0p/chlu/specs/*.json $A/stage1/chlu/specs/
mkdir -p $A/stage1/figures/data && cp $A/stage0p/figures/data/*.json $A/stage1/figures/data/
cd $A/stage1 && /Users/batmac/clu/workspace/.venv/bin/ruff check --select F821,F822,F823 --no-cache --output-format concise . && echo SHAKE-OK $(find . -name '*.py' | xargs cat | wc -l) lines
