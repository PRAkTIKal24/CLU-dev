#!/usr/bin/env bash
# w13 hub/encoder-adapt -- kill -9 RESUMABILITY, PROVED BEFORE ANY SPEND.
#
# The task file: "kill -9 resumability must be PROVED before any spend -- checkpoint on a
# wall-clock timer, verify a config fingerprint before resuming."  This is that proof, and
# it is a measurement, not a description.
#
#   PART 1  a training run is launched with a 5 s checkpoint timer and SIGKILLed mid-flight
#           by THIS script (kill -9 <its own recorded PID>).  Never a box-wide kill:
#           the PID is captured at launch and only that PID is signalled.
#   PART 2  the same command is relaunched.  It must print "RESUMED from checkpoint step N"
#           and run to completion.
#   PART 3  the finished arm is compared against an UNINTERRUPTED run of the identical
#           configuration in a separate directory: the encoder / store / decoder parameter
#           hashes are compared.
#   PART 4  a resume offered a DIFFERENT fingerprint must be REFUSED BY NAME
#           (exp_w13_encoder_adapt --stage killproof).
#
# Every artefact lands in $OUT/killproof/.  Nothing here touches another agent's process,
# worktree or output directory.
set -uo pipefail

OUT="${1:?usage: w13_killproof.sh <out-dir> <seed>}"
SEED="${2:-0}"
WARM="${3:?warm-start run base (the --out that holds seed_<S>/warm_start.json)}"
KP="$OUT/killproof"
mkdir -p "$KP"
PY=(python -u -m chlu.experiments.exp_w13_encoder_adapt)
ARM=adapter                     # the arm with BOTH kinds of leaf: frozen base + trainable adapter
COMMON=(--stage train --seed "$SEED" --arm "$ARM" --quick --ckpt-every-s 5)

note() { echo "[killproof] $*" | tee -a "$KP/killproof.log"; }

note "START $(date -u +%Y-%m-%dT%H:%M:%SZ)  arm=$ARM seed=$SEED"

# The train stage starts from the SHARED warm start, so both proof directories are given
# a byte-for-byte copy of it.  Copying (not re-deriving) is the point: the proof is about
# the training loop's resumability, and a re-derived warm start would change the object.
seed_copy() {
  local dst="$1"
  mkdir -p "$dst/seed_$SEED"
  cp -f "$WARM/seed_$SEED/warm_start.json"   "$dst/seed_$SEED/" || return 1
  cp -f "$WARM/seed_$SEED/warm_encoder.pkl"  "$dst/seed_$SEED/" || return 1
  cp -f "$WARM/seed_$SEED/warm_store.pkl"    "$dst/seed_$SEED/" || return 1
  cp -f "$WARM/seed_$SEED/warm_decoder.pkl"  "$dst/seed_$SEED/" || return 1
  cp -f "$WARM/seed_$SEED/warm_rows.npz"     "$dst/seed_$SEED/" || return 1
}

# ---------------------------------------------------------------- PART 1
rm -rf "$KP/interrupted"
mkdir -p "$KP/interrupted"
seed_copy "$KP/interrupted" || { note "PART 1: FAILED -- no warm start under $WARM"; exit 2; }
note "PART 1: launching, 5 s checkpoint timer, will SIGKILL my own PID only"
"${PY[@]}" "${COMMON[@]}" --out "$KP/interrupted" > "$KP/run_interrupted_1.log" 2>&1 &
VICTIM=$!
echo "$VICTIM" > "$KP/VICTIM_PID.txt"
note "PART 1: victim PID = $VICTIM (recorded in VICTIM_PID.txt)"

# wait until at least one checkpoint exists, then kill -9 THAT PID and nothing else
DEADLINE=$(( $(date +%s) + 900 ))
while :; do
  if ls "$KP"/interrupted/seed_"$SEED"/"$ARM"/ckpt/ckpt_*.npz >/dev/null 2>&1; then break; fi
  if ! kill -0 "$VICTIM" 2>/dev/null; then
    note "PART 1: FAILED -- the run exited before any checkpoint appeared"; break
  fi
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    note "PART 1: FAILED -- no checkpoint within 900 s"; break
  fi
  sleep 2
done
BEFORE=$(ls "$KP"/interrupted/seed_"$SEED"/"$ARM"/ckpt/ckpt_*.npz 2>/dev/null | tail -1 || true)
note "PART 1: newest checkpoint before the kill = ${BEFORE:-<none>}"
if kill -0 "$VICTIM" 2>/dev/null; then
  kill -9 "$VICTIM" 2>/dev/null
  note "PART 1: SIGKILL sent to $VICTIM"
  wait "$VICTIM" 2>/dev/null
  note "PART 1: victim reaped; exit status $?"
else
  note "PART 1: victim already gone"
fi
ls -la "$KP"/interrupted/seed_"$SEED"/"$ARM"/ckpt/ 2>/dev/null | tee -a "$KP/killproof.log"

# ---------------------------------------------------------------- PART 2
note "PART 2: relaunching the IDENTICAL command; it must RESUME"
"${PY[@]}" "${COMMON[@]}" --out "$KP/interrupted" > "$KP/run_resumed.log" 2>&1
RC=$?
note "PART 2: exit $RC"
grep -E "RESUMED from checkpoint step|no checkpoint -- fresh start" "$KP/run_resumed.log" \
  | tee -a "$KP/killproof.log" || note "PART 2: no resume line found"

# ---------------------------------------------------------------- PART 3
note "PART 3: an UNINTERRUPTED run of the identical configuration, for comparison"
rm -rf "$KP/clean"; mkdir -p "$KP/clean"
seed_copy "$KP/clean" || note "PART 3: FAILED -- no warm start to copy"
"${PY[@]}" "${COMMON[@]}" --out "$KP/clean" > "$KP/run_clean.log" 2>&1
note "PART 3: exit $?"

python - "$KP" "$SEED" "$ARM" <<'PYEOF' | tee -a "$KP/killproof.log"
import json, sys
from pathlib import Path
kp, seed, arm = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
def man(which):
    p = kp / which / f"seed_{seed}" / arm / "arm.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
a, b = man("interrupted"), man("clean")
rec = {"resumed_run": None, "clean_run": None, "IDENTICAL": None}
if a and b:
    ha, hb = a["param_hashes"], b["param_hashes"]
    rec = {
        "resumed_run": {"param_hashes": ha,
                        "resumed_from_step": a["compute"]["resumed_from_step"],
                        "checkpoint_saves": a["compute"]["checkpoint_saves"],
                        "fingerprint": a["checkpointing"]["fingerprint"]},
        "clean_run": {"param_hashes": hb,
                      "resumed_from_step": b["compute"]["resumed_from_step"],
                      "fingerprint": b["checkpointing"]["fingerprint"]},
        "fingerprints_equal": a["checkpointing"]["fingerprint"] == b["checkpointing"]["fingerprint"],
        "IDENTICAL": {k: (ha.get(k) == hb.get(k)) for k in ha},
        "note": ("the render decoder is refit from a fresh PRNG split after the training "
                 "loop, so a resumed run's DECODER may differ; what the proof requires is "
                 "that the TRAINED encoder/store/joint_decoder match, which is what the "
                 "checkpoint covers."),
    }
(kp / "killproof.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
print(json.dumps(rec, indent=2))
PYEOF

# ---------------------------------------------------------------- PART 4
note "PART 4: a resume under a DIFFERENT fingerprint must be REFUSED BY NAME"
"${PY[@]}" --stage killproof --seed "$SEED" --arm "$ARM" --out "$OUT" \
  > "$KP/run_fingerprint_refusal.log" 2>&1
tail -3 "$KP/run_fingerprint_refusal.log" | tee -a "$KP/killproof.log"
note "END $(date -u +%Y-%m-%dT%H:%M:%SZ)"
