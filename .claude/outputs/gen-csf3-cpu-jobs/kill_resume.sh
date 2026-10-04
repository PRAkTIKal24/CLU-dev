#!/bin/bash
# ⭐ ACCEPTANCE 3: resume demonstrated by KILLING IT. kill -9 mid-sweep, restart,
# show it SKIPS completed units and resumes — not restarts.
cd /Users/user/Desktop/CHLU-cpujobs
S=scripts/csf3/job_cpu_gen.sh
W=$(mktemp -d)
R="$W/art"; mkdir -p "$R/data/gen" "$R/outputs/gen-feature-bridge/runs" \
    "$R/outputs/gen-store-grouping/runs" "$R/outputs/gen-decoder-lift/winner"
CK="$HOME/scratch/cpugen-killtest"; rm -rf "$CK"

E=(env -i PATH="$PATH" HOME="$HOME" GEN_LOCAL=1 SMOKE=1 SMOKE_SLEEP_S=4
   EXP=atom_mean DATASET=shapes3d PRETEXT=recon SEEDS="0 1 2 3 4"
   CHLU_CLAUDE_ROOT="$R" CKPT_DIR="$CK" CKPT_EVERY_SECONDS=0 AGGREGATE=0)

echo "########## ATTEMPT 1 — will be kill -9'd after ~10s (mid unit 3 of 5) ##########"
"${E[@]}" bash "$S" > "$W/run1.log" 2>&1 &
PID=$!
sleep 10
echo "--- kill -9 of the whole process group (pid $PID + children) ---"
kill -9 -"$PID" 2>/dev/null || pkill -9 -P "$PID" 2>/dev/null; kill -9 "$PID" 2>/dev/null
wait "$PID" 2>/dev/null
echo "attempt-1 exit: $?"
cat "$W/run1.log"
echo
echo "--- journal after the kill ---"
cat "$CK/journal.json"
echo
echo "--- manifests on disk after the kill ---"
ls -1 "$R"/outputs/gen-atom-mean-control/runs/shapes3d_recon/seed_*/atom_manifest.json 2>/dev/null
echo
echo "########## ATTEMPT 2 — same command line, RESUME=1 is the default ##########"
"${E[@]}" bash "$S" 2>&1
echo "attempt-2 exit: $?"
echo
echo "--- final journal ---"
cat "$CK/journal.json"
echo
echo "########## ATTEMPT 3 — fully-resumed run: EVERY unit must SKIP ##########"
"${E[@]}" bash "$S" 2>&1 | grep -E "SKIP|RUN |summary"
echo
echo "########## FINGERPRINT REFUSAL — same CKPT_DIR, different config (QUICK=1) ##########"
env -i PATH="$PATH" HOME="$HOME" GEN_LOCAL=1 SMOKE=1 EXP=atom_mean DATASET=shapes3d \
    PRETEXT=recon SEEDS="0" CHLU_CLAUDE_ROOT="$R" CKPT_DIR="$CK" QUICK=1 AGGREGATE=0 \
    bash "$S" 2>&1 | tail -8
echo "refusal exit: ${PIPESTATUS[0]}"
echo
echo "########## JOURNAL ENTRY WITHOUT ITS MANIFEST — must NOT skip ##########"
rm -f "$R"/outputs/gen-atom-mean-control/runs/shapes3d_recon/seed_1/atom_manifest.json
"${E[@]}" bash "$S" 2>&1 | grep -E "SKIP|RUN |GONE|DONE|summary"
echo
echo "workdir=$W ckpt=$CK"
