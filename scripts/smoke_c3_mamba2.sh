#!/usr/bin/env bash
# scripts/smoke_c3_mamba2.sh — ⭐ THE MAMBA-2 RIVAL ARM'S SMOKE LEG (laptop, minutes).
#
# ############################################################################
# ⛔⛔ THIS IS NEVER A CLAIM VENUE. ⛔⛔
#
# d_model 32, 2 layers, 3 optimisation steps, 0.6 MB of enwik8. The shapes make
# every SEAM execute quickly; they make no number mean anything. A bpc from this
# script is not evidence for or against the CLU or the rival IN EITHER
# DIRECTION — including the case where Mamba-2 wins here. Charter §2: claims live
# at 26-47 M on CSF3 with >=3 seeds. If you are about to quote a number from this
# script, stop.
# ############################################################################
#
# WHAT IT PROVES (task §1.6, in order):
#   the mamba2 arm trains -> checkpoints -> RESUMES -> evaluates -> emits slices
#   -> and LEDGERS, reproducing RIVAL_SPECS' pinned value to the byte.
#
#   leg 1  fresh run, interrupted (os._exit) right after the mamba2 arm banks
#   leg 2  --resume: the banked mamba2 checkpoint is lifted, the run completes
#   leg 3  --resume --slices on the finished leg: only the slice phases run
#   then   the acceptance checks, including the byte-exact reference reproduction
#
# ⛔ `clu_store` is in --arms because the pilot's S1/S2 phases index
#    models["clu_store"] unconditionally; a rival-ONLY job cannot run today.
#    Reported as a finding, not worked around silently.
#
# USAGE:  bash scripts/smoke_c3_mamba2.sh [OUT_DIR] [CORPUS]
# ⚠ enwik8 must already be staged (~36 MB, once):
#       python -c "from chlu.data.corpora import stage_corpus; stage_corpus('enwik8')"

set -euo pipefail

OUT="${1:-.claude/outputs/c3-rival-mamba2/smoke}"
CORPUS="${2:-enwik8}"
PY="${PY:-python}"

rm -rf "$OUT"
mkdir -p "$OUT"

echo "############ leg 1/3 — fresh run, INTERRUPTED after the mamba2 arm ############"
set +e
$PY -u -m chlu.experiments.exp_cluformer_pilot \
    --scale toy --stage s3 --seed 0 --out "$OUT" --corpus "$CORPUS" \
    --arms mamba2 clu_store \
    --set d_model=32 n_layers=2 seq_len=256 batch=2 steps=3 warmup=1 \
          eval_batches=2 dyneval_batches=2 slice_batches=2 slice_min_n=5 \
          data_bytes=600000 monitor_every=1 stop_after_arms=1 \
    --mem chunk=32 address_steps=4 read_steps=4 traj_stride=2 psi_hidden=16 \
          write_inner_steps=1 write_n_perturb=4
rc=$?
set -e
[ "$rc" -eq 137 ] || { echo "⛔ leg 1 should have hard-exited 137, got $rc"; exit 1; }
test -f "$OUT/pilot_toy_seed0_PARTIAL.json" || { echo "⛔ no journal"; exit 1; }
test -f "$OUT/ckpt_mamba2_seed0.eqx"        || { echo "⛔ no mamba2 checkpoint"; exit 1; }
echo "✓ leg 1: the rival arm trained, banked and checkpointed"

echo "############ leg 2/3 — RESUME: lift the banked rival, finish the run ##########"
$PY -u -m chlu.experiments.exp_cluformer_pilot \
    --scale toy --stage s3 --seed 0 --out "$OUT" --corpus "$CORPUS" --resume \
    --arms mamba2 clu_store \
    --set d_model=32 n_layers=2 seq_len=256 batch=2 steps=3 warmup=1 \
          eval_batches=2 dyneval_batches=2 slice_batches=2 slice_min_n=5 \
          data_bytes=600000 monitor_every=1 \
    --mem chunk=32 address_steps=4 read_steps=4 traj_stride=2 psi_hidden=16 \
          write_inner_steps=1 write_n_perturb=4
test -f "$OUT/pilot_toy_seed0_S3.json" || { echo "⛔ no final artifact"; exit 1; }
echo "✓ leg 2: resumed to a final artifact"

echo "############ leg 3/3 — RESUME + --slices on the FINISHED leg ##################"
$PY -u -m chlu.experiments.exp_cluformer_pilot \
    --scale toy --stage s3 --seed 0 --out "$OUT" --corpus "$CORPUS" \
    --resume --slices \
    --arms mamba2 clu_store \
    --set d_model=32 n_layers=2 seq_len=256 batch=2 steps=3 warmup=1 \
          eval_batches=2 dyneval_batches=2 slice_batches=2 slice_min_n=5 \
          data_bytes=600000 monitor_every=1 \
    --mem chunk=32 address_steps=4 read_steps=4 traj_stride=2 psi_hidden=16 \
          write_inner_steps=1 write_n_perturb=4
test -f "$OUT/slices_toy_seed0.json" || { echo "⛔ no slice artifact"; exit 1; }
echo "✓ leg 3: slices added to a finished leg by resume alone"

echo "############ the acceptance checks ############"
$PY - "$OUT" <<'PY'
import json, math, sys, pathlib
from chlu.eval.byte_ledger import RIVAL_SPECS, shrink_to_budget
from chlu.eval.rivals.c3_mamba2 import Mamba2ArmConfig, PUBLISHED_D_STATE

out = pathlib.Path(sys.argv[1])
rec = json.loads((out / "pilot_toy_seed0_S3.json").read_text())
sl = json.loads((out / "slices_toy_seed0.json").read_text())

# 1. the arm is IN the byte ledger, and the ledger is enforced
bl = rec["byte_ledger"]
assert bl["enforced"] and bl["phi_accounted_on_every_arm"]
row = bl["arms"]["mamba2"]
assert row["within_budget"], row
print(f"byte ledger: mamba2 {row['total_state_bytes']:,} B "
      f"({row['occupancy']:.5f}x of {bl['budget_bytes']:,} B), φ accounted")

# 2. ⭐ THE ACCEPTANCE CRITERION: the pinned config reproduces RIVAL_SPECS TO THE BYTE
led = rec["swap_ledger"]["mamba2"]
cfg = Mamba2ArmConfig(**led["rival_config"])
pub = cfg._replace(d_state=PUBLISHED_D_STATE)
assert pub.total_state_bytes() == RIVAL_SPECS["mamba2"].state_bytes() == 6_475_776, pub
sh = shrink_to_budget("mamba2")
assert cfg.d_state == sh["shrunk_value"] == 39
assert cfg.total_state_bytes() == sh["state_bytes_shrunk"] == 2_075_616
print(f"pinned  24 L @ d_state {PUBLISHED_D_STATE}: {pub.total_state_bytes():,} B "
      f"= RIVAL_SPECS to the byte")
print(f"shrunk  24 L @ d_state {cfg.d_state}: {cfg.total_state_bytes():,} B "
      f"(knob solved by shrink_to_budget, not by hand)")
print(f"deployed {led['deployed']['deployed_n_layers']} L: "
      f"{led['deployed']['total_state_bytes']:,} B "
      f"occupancy {led['deployed']['occupancy']:.4f}")

# 3. no library default was inherited: every state-bearing number has provenance
for k, v in led["provenance"].items():
    assert v.startswith(("PAPER:", "OFFICIAL IMPLEMENTATION:")), (k, v[:40])
assert led["dtype_bytes"] == 2, "dtype must be declared, as deployed (bf16)"
print(f"provenance: {len(led['provenance'])} pinned numbers, all "
      "PAPER:/OFFICIAL IMPLEMENTATION:")

# 4. it trained, and it did not diverge
h = rec["arms"]["mamba2"]["train"]["loss_history"]
assert all(math.isfinite(x) for x in h), h
assert math.isfinite(rec["arms"]["mamba2"]["static"]["bpc"])
assert "dyneval" in rec["arms"]["mamba2"]
print(f"train: {len(h)} finite steps; static+dyn-eval columns present "
      "(⛔ values NOT quotable — this is not a claim venue)")

# 5. it emits retention slices, like every other arm
s = sl["arms"]["mamba2"]["static"]
assert s["n_scored"] > 0 and "dyneval" in sl["arms"]["mamba2"]
print(f"slices: {sum(1 for b in s['bins'].values() if b['bpc'] is not None)}/"
      f"{len(s['bins'])} bins scored, n_scored {s['n_scored']}")

print("\n✅ MAMBA-2 SMOKE PASSED — trains, checkpoints, resumes, evaluates, "
      "slices, ledgers to the byte.")
print("⛔ Reminder: these numbers are NOT a result. This is not a claim venue.")
PY
