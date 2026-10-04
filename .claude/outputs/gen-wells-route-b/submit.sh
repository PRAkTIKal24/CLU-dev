# ⭐ gen-wells-route-b — CSF3 SUBMIT BLOCK for the Head (copy-able, one leg per line)
# ⛔ EVERY LINE BELOW IS **NOT RUN**: CSF3 needs 2FA and the agent cannot reach it.
# ⚠ Run these in order. Legs 0–2 are one-time-per-push; leg 3 is the long job.

# ---- 0. LAPTOP (on campus / GlobalProtect VPN): ship code + lockfile ----------
# ⛔ `git pull` on the cluster serves a STALE fork — push_repo.sh is the ONLY path.
export CSF3_USER=<uom-username> && cd /Users/user/Desktop/CHLU && uv lock && scripts/csf3/push_repo.sh

# ---- 1. LAPTOP: ship the gen artefact set (⛔ owned by gen-cluster-stage-artefacts) ----
# ⚠ BLOCKER, see the report §"What is NOT run": STAGED.json stages 44 files / 1.95 GiB but
# under data/gen/ it stages ONLY shapes3d/c_invalid.json. `3dshapes.h5` (255 MB) is NOT in
# the set and this job needs it (factor labels + the cache re-encode check).
# ⛔ Do NOT improvise a staging path — hand this back to that spoke first.
scripts/csf3/stage_artefacts.sh            # ⛔ NOT MINE; confirm 3dshapes.h5 is included

# ---- 2. CSF3 LOGIN NODE: build the env on a COMPUTE node (login nodes have no PyPI) ----
ssh $CSF3_USER@csf3.itservices.manchester.ac.uk
cd ~/scratch/CHLU && sbatch scripts/csf3/setup_env_job.sh          # ~5–15 min queued+built

# ---- 3. ⭐ THE LONG JOB (login node). GEN_GIT_COMMIT is exported HERE, on the login node ----
cd ~/scratch/CHLU && mkdir -p logs && export GEN_GIT_COMMIT=$(git -C ~/scratch/CHLU rev-parse HEAD 2>/dev/null || echo rsync-no-git)

# 3a. plumbing smoke first (~30 min, ⛔ no number from it is ever reported)
sbatch -t 0-4 --export=ALL,GEN_GIT_COMMIT,QUICK=1,SCORED=0 scripts/csf3/job_gpu_wells.sh

# 3b. ⭐ THE SCORED RUN — seed 0 only (⚠ seeds 1/2 hold 9,288 rows and the coverage
#     assertion will name the shortfall and die at startup, by design).
sbatch -t 2-0 --export=ALL,GEN_GIT_COMMIT,SEEDS=0,SCORED=1 scripts/csf3/job_gpu_wells.sh

# 3c. ⭐⭐ THE RESUME COMMAND — identical, plus RESUME=1. Safe to run at any time: a
#     checkpoint written under a different config is REFUSED, not silently blended.
sbatch -t 2-0 --export=ALL,GEN_GIT_COMMIT,SEEDS=0,SCORED=1,RESUME=1 scripts/csf3/job_gpu_wells.sh

# ---- 4. monitor / pull ------------------------------------------------------
squeue --me                                   # ST=PD pending, R running
sacct -j <jobid> --format=JobID,Elapsed,State,MaxRSS
scancel <jobid>
ls -1 ~/scratch/CHLU/ckpt/gen-wells-route-b/seed_0/     # the banked checkpoints
# LAPTOP: pull the artefacts back (⛔ a result that is not synced back does not exist)
rsync -av $CSF3_USER@csf3.itservices.manchester.ac.uk:'~/scratch/CHLU/artefacts/outputs/gen-wells-route-b/' /Users/user/Desktop/CHLU/.claude/outputs/gen-wells-route-b/cluster/

# ---- expected wall time (⚠ EXTRAPOLATED from a CPU verification run, not measured on an A100) ----
#   3a  QUICK smoke .................... ~0.5 h   (-t 0-4 is generous headroom)
#   3b  scored, seed 0, 4 arms ......... ~6–10 h  (-t 2-0 = 48 h, ~5x headroom)
#       ⚠ The headroom is deliberate: the A100 figure is an extrapolation, and a too-small
#       -t gets the job SIGTERM'd (30 s grace, then SIGKILL). With checkpointing a
#       generous -t costs nothing on a free-at-point-of-use partition.
# ---- cost ----
#   CSF3 gpuA A100-SXM4-80GB: FREE AT POINT OF USE  =>  £0 / $0.
#   ⚠ ⛔ This is NOT the GCP A100-80GB that PREREG §12 records as quota-zero for this
#   project. They are different resources and neither line may be quoted as the other.
