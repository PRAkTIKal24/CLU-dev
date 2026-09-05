#!/usr/bin/env bash
# Demo 2 (retry, lighter): the OVER-BUDGET branch end to end THROUGH A TRAINING RUN.
# ⚠ The first attempt at batch=4/seq_len=512 was SIGKILLed at 3.3 GB RSS in
# allocation_liveness (laptop memory, not a code defect) — so batch=1,
# seq_len=128, liveness_lanes=1. ⛔ The GEOMETRY (payload_dim=300 => 2.51 MB)
# is what busts the shipped 2 MiB budget; the budget flag is untouched.
# ⚠ DISCLOSED: the run-2 journal is SYNTHETIC (flag_block output, C2W6 keys
# popped as the real ones have them) — an over-budget leg cannot be produced by
# current code, which is exactly why only the pre-guard real run 2 has one.
set -u
cd /Users/user/Desktop/CHLU-wt1
export PYTHONPATH=/Users/user/Desktop/CHLU-wt1
PY=/Users/user/Desktop/CHLU/.venv/bin/python
D=/Users/user/Desktop/CHLU/.claude/scratch/c3-run3-budget-exemption
rm -rf "$D/run2d" "$D/run3d"; mkdir -p "$D/run2d"

$PY - <<'EOF'
import json
from pathlib import Path
import chlu.experiments.exp_cluformer_pilot as EXP
D = Path("/Users/user/Desktop/CHLU/.claude/scratch/c3-run3-budget-exemption")
cfg = EXP.make_config("toy", 0, {
    "steps": 4, "warmup": 1, "eval_batches": 1, "dyneval_batches": 1,
    "data_bytes": 300000, "monitor_every": 2, "payload_dim": 300,
    "batch": 1, "seq_len": 128, "liveness_lanes": 1,
    "arms": ("clu_store", "none")})
flags = EXP.flag_block(cfg)
for k in ("erosion_partition", "refresh_amp_ceiling", "refresh_max_gain",
          "refresh_monotonic"):
    flags["memory"].pop(k)
p = EXP.partial_path(D / "run2d", "toy", 0)
p.write_text(json.dumps({"flags": flags, "arms": {}, "_journal": {"trained": {}}}))
print("synthetic run-2 journal:", p)
EOF

echo "=== A) the run-3 config, over budget, WITHOUT the exemption ============="
$PY -m chlu.experiments.exp_cluformer_pilot --scale toy --stage s3 --seed 0 --out "$D/run3d" --arms clu_store none --set steps=4 warmup=1 eval_batches=1 dyneval_batches=1 data_bytes=300000 monitor_every=2 payload_dim=300 batch=1 seq_len=128 liveness_lanes=1 --mem erosion_partition=true > "$D/legA.log" 2>&1
echo "EXIT_A=$?"

echo "=== B) the same run, WITH the exemption ================================="
$PY -m chlu.experiments.exp_cluformer_pilot --scale toy --stage s3 --seed 0 --out "$D/run3d" --arms clu_store none --set steps=4 warmup=1 eval_batches=1 dyneval_batches=1 data_bytes=300000 monitor_every=2 payload_dim=300 batch=1 seq_len=128 liveness_lanes=1 --mem erosion_partition=true --prereg-continuation journal="$D/run2d/pilot_toy_seed0_PARTIAL.json" flag=memory.erosion_partition prereg=.claude/outputs/c2w6-anti-erosion/PREREG-LeakAblation.md > "$D/legB.log" 2>&1
echo "EXIT_B=$?"
ls -la "$D/run3d"
