#!/bin/bash
# Every guard in job_cpu_gen.sh, driven to FIRE. Run from the worktree root.
S=scripts/csf3/job_cpu_gen.sh
run() { # $1 = label, rest = VAR=VAL ...
  label="$1"; shift
  echo "######## $label ########"
  env -i PATH="$PATH" HOME="$HOME" GEN_LOCAL=1 EXP=atom_mean DATASET=shapes3d \
      PRETEXT=recon PLAN=1 CHLU_SKIP_ARTEFACT_CHECK=1 "$@" bash "$S" 2>&1 | tail -8
  echo "rc=${PIPESTATUS[0]}"; echo
}
run "G1  -c 96 (over the 84 cap)"        SLURM_CPUS_PER_TASK=96
run "G1b -c 84 (AT the cap: must pass)"  SLURM_CPUS_PER_TASK=84
run "G1c -c 0"                           SLURM_CPUS_PER_TASK=0
run "G1d -c notanumber"                  SLURM_CPUS_PER_TASK=eighty
run "G2  account=gpu-free"               SLURM_JOB_ACCOUNT=gpu-free
run "G2b account=fatpou01 (must pass)"   SLURM_JOB_ACCOUNT=fatpou01
run "G3  partition=serial"               SLURM_JOB_PARTITION=serial
run "G4  EXP unset"                      EXP=
run "G4b EXP=nonesuch"                   EXP=nonesuch
run "G4c DATASET unset"                  DATASET=
run "G4d PRETEXT=bogus"                  PRETEXT=bogus
run "G4e ARM=bogus"                      ARM=bogus EXP=baselines
run "G4f SEED non-integer"               SEEDS="0 x"

echo "######## G5 missing artefact root (named) ########"
env -i PATH="$PATH" HOME="$HOME" GEN_LOCAL=1 EXP=atom_mean DATASET=shapes3d PRETEXT=recon \
    CLU_REPO=/tmp/cpugen-no-artefacts bash "$S" 2>&1 | tail -8
echo "rc=$?"; echo

echo "######## G5b roots present except ONE (named) ########"
R=$(mktemp -d)/art
mkdir -p "$R/data/gen" "$R/outputs/gen-feature-bridge/runs" "$R/outputs/gen-store-grouping/runs"
env -i PATH="$PATH" HOME="$HOME" GEN_LOCAL=1 EXP=atom_mean DATASET=shapes3d PRETEXT=recon \
    CHLU_CLAUDE_ROOT="$R" bash "$S" 2>&1 | tail -6
echo "rc=$?"; echo

echo "######## G6 CKPT_DIR under \$TMPDIR ########"
env -i PATH="$PATH" HOME="$HOME" GEN_LOCAL=1 EXP=atom_mean DATASET=shapes3d PRETEXT=recon \
    CHLU_SKIP_ARTEFACT_CHECK=1 TMPDIR=/tmp/xyz CKPT_DIR=/tmp/xyz/ckpt bash "$S" 2>&1 | tail -3
echo "rc=$?"; echo

echo "######## G6b CKPT_DIR under \$HOME (not ~/scratch) ########"
env -i PATH="$PATH" HOME="$HOME" GEN_LOCAL=1 EXP=atom_mean DATASET=shapes3d PRETEXT=recon \
    CHLU_SKIP_ARTEFACT_CHECK=1 CKPT_DIR="$HOME/ckpt" bash "$S" 2>&1 | tail -3
echo "rc=$?"; echo
