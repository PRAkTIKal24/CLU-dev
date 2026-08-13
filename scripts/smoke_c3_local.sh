#!/usr/bin/env bash
# scripts/smoke_c3_local.sh — ⭐ THE C3 HARNESS SMOKE CONFIG (laptop, minutes).
#
# ############################################################################
# ⛔⛔ THIS IS NEVER A CLAIM VENUE. ⛔⛔
#
# The shapes below (d_model 32, 2 layers, 3 optimisation steps, 0.6 MB of
# enwik8) are chosen to make every SEAM execute quickly, not to make any number
# mean anything. A bpc from this config is not evidence for or against the CLU,
# a rival, or anything else, IN EITHER DIRECTION — including the case where the
# TTT arm beats the CLU here. Charter §2: claims live at 26-47 M on CSF3 with
# >=3 seeds and the tier's own control. If you are about to quote a number that
# came out of this script, stop.
# ############################################################################
#
# WHAT IT PROVES (the acceptance path, in order):
#   load -> train -> checkpoint -> RESUME -> eval -> slices -> byte ledger
#
# It runs the ladder the cluster runs, in miniature:
#   leg 1  a fresh run WITHOUT slices, interrupted after the first arm
#          (stop_after_arms=1 hard-exits like an oom_kill: no finalisers)
#   leg 2  --resume: the banked arm is lifted, the rest completes
#   leg 3  --resume --slices on the FINISHED leg: because --slices is a CLI
#          argument and NOT a PilotConfig field it cannot move the resume
#          fingerprint, so ONLY the slice phases run and everything else is
#          lifted verbatim. That is the resume-first ladder in one command.
#
# USAGE:  bash scripts/smoke_c3_local.sh [OUT_DIR] [CORPUS]
#   CORPUS is a config value: enwik8 (default) | wikitext103.
#
# ⚠ JAX cold start on the authoring laptop is ~15 s warm but can be ~20 min
#   cold, even for --help. Budget for it; it is not a hang.
# ⚠ enwik8 must already be staged (~36 MB download, once):
#       python -c "from chlu.data.corpora import stage_corpus; stage_corpus('enwik8')"

set -euo pipefail

OUT="${1:-.claude/outputs/c3-csf3-harness/smoke}"
CORPUS="${2:-enwik8}"
PY="${PY:-python}"

rm -rf "$OUT"
mkdir -p "$OUT"

echo "############ leg 1/3 — fresh run, INTERRUPTED after one arm ############"
# `stop_after_arms=1` hard-exits via os._exit(137) with the journal on disk, so
# leg 2 is gated by an actual interrupted run rather than by inspection.
# ⚠ ONE LITERAL COMMAND LINE. zsh does not word-split, so an args-in-a-variable
#   refactor silently submits garbage (task §3.4 — this bit the author live).
set +e
$PY -u -m chlu.experiments.exp_cluformer_pilot \
    --scale toy --stage s3 --seed 0 --out "$OUT" --corpus "$CORPUS" \
    --arms clu_store none \
    --set d_model=32 n_layers=2 seq_len=256 batch=2 steps=3 warmup=1 \
          eval_batches=2 dyneval_batches=2 slice_batches=2 slice_min_n=5 \
          data_bytes=600000 monitor_every=1 stop_after_arms=1 \
    --mem chunk=32 address_steps=4 read_steps=4 traj_stride=2 psi_hidden=16 \
          write_inner_steps=1 write_n_perturb=4
rc=$?
set -e
[ "$rc" -eq 137 ] || { echo "⛔ leg 1 should have hard-exited 137, got $rc"; exit 1; }
test -f "$OUT/pilot_toy_seed0_PARTIAL.json" || { echo "⛔ no journal"; exit 1; }
test -f "$OUT/ckpt_clu_store_seed0.eqx"     || { echo "⛔ no checkpoint"; exit 1; }
echo "✓ leg 1: journal + checkpoint on disk after a simulated kill"

echo "############ leg 2/3 — RESUME: lift the banked arm, finish the run ######"
$PY -u -m chlu.experiments.exp_cluformer_pilot \
    --scale toy --stage s3 --seed 0 --out "$OUT" --corpus "$CORPUS" --resume \
    --arms clu_store none \
    --set d_model=32 n_layers=2 seq_len=256 batch=2 steps=3 warmup=1 \
          eval_batches=2 dyneval_batches=2 slice_batches=2 slice_min_n=5 \
          data_bytes=600000 monitor_every=1 \
    --mem chunk=32 address_steps=4 read_steps=4 traj_stride=2 psi_hidden=16 \
          write_inner_steps=1 write_n_perturb=4
test -f "$OUT/pilot_toy_seed0_S3.json" || { echo "⛔ no final artifact"; exit 1; }
echo "✓ leg 2: resumed to a final artifact"

echo "############ leg 3/3 — RESUME + --slices on the FINISHED leg ############"
$PY -u -m chlu.experiments.exp_cluformer_pilot \
    --scale toy --stage s3 --seed 0 --out "$OUT" --corpus "$CORPUS" \
    --resume --slices \
    --arms clu_store none \
    --set d_model=32 n_layers=2 seq_len=256 batch=2 steps=3 warmup=1 \
          eval_batches=2 dyneval_batches=2 slice_batches=2 slice_min_n=5 \
          data_bytes=600000 monitor_every=1 \
    --mem chunk=32 address_steps=4 read_steps=4 traj_stride=2 psi_hidden=16 \
          write_inner_steps=1 write_n_perturb=4
test -f "$OUT/slices_toy_seed0.json" || { echo "⛔ no slice artifact"; exit 1; }
echo "✓ leg 3: slices added to a finished leg by resume alone"

echo "############ the acceptance checks ############"
$PY - "$OUT" <<'PY'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
rec = json.loads((out / "pilot_toy_seed0_S3.json").read_text())
sl = json.loads((out / "slices_toy_seed0.json").read_text())

bl = rec["byte_ledger"]
assert bl["phi_accounted_on_every_arm"], "φ is not accounted on every arm"
assert bl["enforced"], "the state-byte budget was not enforced"
print(f"byte ledger: budget {bl['budget_bytes']:,} B, enforced, φ on every arm")
for a, r in sorted(bl["arms"].items()):
    print(f"   {a:12s} {r['total_state_bytes']:>10,} B  occupancy {r['occupancy']:.5f}"
          f"  within={r['within_budget']}")
assert bl["rival_reference"]["rivals"], "no pinned rival reference table"

c = sl["controls"]
assert c["non_degeneracy"]["passed"], "TRAP 1: the slice is a frequency count"
assert c["shuffled_position"]["slice_moved"], "the slice did not move under shuffling"
assert c["content_relabel"]["slice_invariant"], "the slice moved under relabelling"
print(f"slice controls: non-degeneracy ratio "
      f"{c['non_degeneracy']['ratio']:.1f}x (token vs raw-byte unit), "
      f"shuffle TVD {c['shuffled_position']['total_variation_distance']:.4f}, "
      f"relabel invariant {c['content_relabel']['slice_invariant']}")
for arm, s in sorted(sl["arms"].items()):
    st = s["static"]
    filled = sum(1 for b in st["bins"].values() if b["bpc"] is not None)
    print(f"   {arm:12s} slices: {filled}/{len(st['bins'])} bins scored, "
          f"n_scored {st['n_scored']}, dyn-eval column present="
          f"{'dyneval' in s}")
    assert "dyneval" in s, "the dyn-eval substitute column is missing from the slice"
print("\n✅ SMOKE PASSED — load, train, checkpoint, resume, eval, slices, ledger.")
print("⛔ Reminder: these numbers are NOT a result. This is not a claim venue.")
PY
