#!/usr/bin/env bash
# c3-run3-budget-exemption — the end-to-end demonstration.
# ⛔ NEVER A CLAIM VENUE: toy scale, 4 steps. Nothing here is a result.
# ⚠ zsh has no word-splitting: ONE LITERAL COMMAND LINE PER RUN. No $CFG vars.
set -u
cd /Users/user/Desktop/CHLU-wt1
export PYTHONPATH=/Users/user/Desktop/CHLU-wt1
PY=/Users/user/Desktop/CHLU/.venv/bin/python
D=/Users/user/Desktop/CHLU/.claude/scratch/c3-run3-budget-exemption
rm -rf "$D/run2" "$D/run3" "$D/run2b" "$D/run3b"

echo "=== LEG 1: 'run 2' — a REAL toy leg, shield OFF, default budget ==========="
$PY -m chlu.experiments.exp_cluformer_pilot --scale toy --stage s3 --seed 0 --out "$D/run2" --arms clu_store none --set steps=4 warmup=1 eval_batches=1 dyneval_batches=1 data_bytes=300000 monitor_every=2 2>&1 | tail -12

echo "=== LEG 2: 'run 3' = leg 1 + erosion_partition=True, under the exemption =="
$PY -m chlu.experiments.exp_cluformer_pilot --scale toy --stage s3 --seed 0 --out "$D/run3" --arms clu_store none --set steps=4 warmup=1 eval_batches=1 dyneval_batches=1 data_bytes=300000 monitor_every=2 --mem erosion_partition=true --prereg-continuation journal="$D/run2/pilot_toy_seed0_PARTIAL.json" flag=memory.erosion_partition prereg=.claude/outputs/c2w6-anti-erosion/PREREG-LeakAblation.md 2>&1 | tail -20
