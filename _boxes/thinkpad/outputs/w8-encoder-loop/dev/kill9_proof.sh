#!/usr/bin/env bash
# w8 hub/encoder-loop -- THE kill -9 RESUMABILITY PROOF (standing Head rule).
#
# A spot VM is taken away mid-step.  The claim this proof has to earn is:
#
#     killing the training process with SIGKILL costs the steps since the last
#     checkpoint AND NOTHING ELSE -- the relaunch RESUMES, it does not restart,
#     and the resumed run reaches the SAME PARAMETERS as an uninterrupted one.
#
# It is proved by BYTES, not by a log line: the encoder's parameter hash of the
# killed-and-resumed run is compared against the parameter hash of a clean run of
# the identical configuration.  Equal hashes = resumed.  Different = restarted, or
# resumed wrong, and the spec does not ship.
#
# It also proves the REFUSAL: a checkpoint written under one config fingerprint is
# refused by name when a different configuration tries to resume from it.  A check
# that cannot fail proves nothing.
#
#   bash dev/kill9_proof.sh <out-base>
set -uo pipefail

OUT="${1:-/c/chlu/outputs/w8-encoder-loop/dev/kill9}"
M=chlu.experiments.exp_w8_encoder_loop
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONIOENCODING=utf-8
: "${CHLU_GEN_CLASSIFIER_ROOT:=$(python -c "import os,pathlib;print(pathlib.Path(os.environ['CHLU_GEN_BRIDGE_ROOT']).parent.parent / 'gen-external-classifier')")}"
export CHLU_GEN_CLASSIFIER_ROOT

rm -rf "$OUT"; mkdir -p "$OUT"
echo "== kill -9 proof, out=$OUT"

# ---------------------------------------------------------------- 0. warm start
# Shared by both legs, so the two runs differ ONLY in being interrupted.
python -u -m "$M" --stage warm --seed 0 --out "$OUT" --quick > "$OUT/warm.log" 2>&1 || {
  echo "!! warm failed"; tail -20 "$OUT/warm.log"; exit 1; }
echo "-- warm start done"

# The two legs share one warm start: copy the seed dir so each has its own arm tree.
CLEAN="$OUT/clean"; KILLED="$OUT/killed"
for d in "$CLEAN" "$KILLED"; do
  mkdir -p "$d"; cp -r "$OUT/seed_0" "$d/seed_0"
  rm -rf "$d/seed_0"/alone "$d/seed_0"/frozen "$d/seed_0"/loop_nosw "$d/seed_0"/loop_swap
done

ARM=loop_swap   # the treatment: the most expensive step, the one a pre-emption will hit

# ------------------------------------------------- 1. the CLEAN, uninterrupted run
echo "-- leg A: clean run (no interruption)"
python -u -m "$M" --stage train --seed 0 --arm "$ARM" --out "$CLEAN" --quick \
  --ckpt-every-s 3 > "$CLEAN/train.log" 2>&1
rcA=$?
echo "   leg A rc=$rcA"

# ------------------------------- 2. the KILLED run: SIGKILL, then relaunch to finish
echo "-- leg B: start, wait for a checkpoint, kill -9, relaunch"
python -u -m "$M" --stage train --seed 0 --arm "$ARM" --out "$KILLED" --quick \
  --ckpt-every-s 3 > "$KILLED/train1.log" 2>&1 &
PID=$!
echo "   leg B pid $PID (the ONLY pid this script ever kills)"

# Wait for at least TWO checkpoints so the resume is from a real mid-run state,
# not from step 0.  Bounded, so a stall fails the proof instead of hanging it.
for _ in $(seq 1 240); do
  n=$(ls "$KILLED/seed_0/$ARM/ckpt"/ckpt_*.npz 2>/dev/null | wc -l)
  [ "${n:-0}" -ge 2 ] && break
  kill -0 "$PID" 2>/dev/null || break
  sleep 2
done
n=$(ls "$KILLED/seed_0/$ARM/ckpt"/ckpt_*.npz 2>/dev/null | wc -l)
echo "   $n checkpoint(s) on disk"
if ! kill -0 "$PID" 2>/dev/null; then
  echo "!! leg B finished before it could be killed -- the proof needs a LIVE kill"
  tail -20 "$KILLED/train1.log"; exit 1
fi
kill -9 "$PID"
wait "$PID" 2>/dev/null
echo "   SIGKILLed $PID"
grep -c "checkpoint @ step" "$KILLED/train1.log" || true

echo "-- leg B: relaunch (must RESUME, not restart)"
python -u -m "$M" --stage train --seed 0 --arm "$ARM" --out "$KILLED" --quick \
  --ckpt-every-s 3 > "$KILLED/train2.log" 2>&1
rcB=$?
echo "   leg B relaunch rc=$rcB"
grep -E "RESUMED from checkpoint|no checkpoint" "$KILLED/train2.log" || true

# ------------------------------------------------------ 3. the verdict, in BYTES
python - "$CLEAN" "$KILLED" "$ARM" <<'PY'
import json, sys, pathlib
clean, killed, arm = (pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3])
a = json.load(open(clean / "seed_0" / arm / "arm.json"))
b = json.load(open(killed / "seed_0" / arm / "arm.json"))
ha, hb = a["param_hashes"]["encoder"], b["param_hashes"]["encoder"]
sa, sb = a["param_hashes"]["store"], b["param_hashes"]["store"]
res = b["compute"]["resumed_from_step"]
print()
print("== VERDICT")
print(f"   clean  encoder {ha[:24]}  store {sa[:24]}")
print(f"   killed encoder {hb[:24]}  store {sb[:24]}")
print(f"   the killed leg RESUMED FROM STEP {res} (0 would mean it restarted)")
ok = (ha == hb) and (sa == sb) and res > 0
print(f"   RESUMABILITY: {'PROVED -- bit-identical parameters after a kill -9' if ok else 'FAILED'}")
json.dump({
    "proof": "kill -9 mid-train, relaunch, compare parameters by hash",
    "arm": arm,
    "clean_encoder_param_hash": ha, "killed_encoder_param_hash": hb,
    "clean_store_param_hash": sa, "killed_store_param_hash": sb,
    "resumed_from_step": res,
    "bit_identical": bool(ha == hb and sa == sb),
    "verdict": "PROVED" if ok else "FAILED",
    "note": ("a resumed run that reaches the SAME PARAMETERS as an uninterrupted one "
             "has resumed; one that reaches different parameters has restarted or "
             "resumed wrong, and either way the spec does not ship"),
}, open(pathlib.Path(sys.argv[2]).parent / "KILL9-PROOF.json", "w"), indent=2)
sys.exit(0 if ok else 1)
PY
V=$?

# ------------------------------------- 4. the REFUSAL: a check that cannot fail proves nothing
echo
echo "-- fingerprint refusal: resume a checkpoint under a DIFFERENT config"
python - "$KILLED" "$ARM" <<'PY'
import importlib.util, pathlib, sys
here = pathlib.Path("scripts/cloud/ckpt.py").resolve()
spec = importlib.util.spec_from_file_location("ck", here)
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
d = pathlib.Path(sys.argv[1]) / "seed_0" / sys.argv[2] / "ckpt"
wrong = C.Checkpointer(str(d), "deadbeefdeadbeef", remote_uri=None, every_s=1e9)
try:
    wrong.latest()
    print("   REFUSAL FAILED: a foreign fingerprint was accepted")
    sys.exit(1)
except C.FingerprintMismatch as e:
    print(f"   REFUSED, by name: {str(e)[:150]}")
    sys.exit(0)
PY
R=$?
echo
echo "== kill -9 proof: resume=$([ $V = 0 ] && echo PROVED || echo FAILED) refusal=$([ $R = 0 ] && echo PROVED || echo FAILED)"
[ $V = 0 ] && [ $R = 0 ]
