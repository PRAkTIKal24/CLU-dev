# RUN3-LAUNCH — the copy-verbatim submission package for CSF3 run 3

**Filed by the C3W1 Hub, 2026-08-13 · REVISED twice same day (v3).** Run 3 = **run 2 + `erosion_partition=true`**, **exactly one flag**, geometry unchanged (`PREREG-LeakAblation.md` §4). Carries **D5** and the **pre-registered-continuation exemption**.

> ## ⛔⛔ v3 — THE MEMORY FIX. v2's LINES WOULD HAVE OOM-KILLED ALL THREE JOBS.
> v2 set `-t` but **not `-G`/`-c`**, so it inherited the script defaults **`-G 1 -c 8`**. On gpuA that is **10 GB host RAM per core ⇒ 80 GB**.
> **The measured run peak is 219.08 GB** — the `dyneval` phase (`+215.75 GB`, 61,686 s), which the `pilot-ttt` report records as *"the phase that demanded 219 GB and **killed four legs**"*, and which job `18136619` hit as an `oom_kill` at **MaxRSS 125.6 GB vs ReqMem 125.7 GB**. **80 GB against a 219 GB peak fails, three times, hours in.**
> ⇒ **All three lines now carry `-G 2 -c 24`** — the **proven-good recovery configuration** the 219.08 GB peak was actually measured under (24 cores × 10 GB ≈ 240 GB; the peak is 87.3 % of the ~251 GB envelope). This is inside the ratified per-job envelope (**2×A100, 4-day**) and changes **no config value**, so the resume fingerprint and the exemption are untouched.
> ⚠ **Do not "optimise" `-G 2` down to `-G 1` because only one GPU is busy.** On gpuA host RAM is bought **per core**, cores are capped at **12 per GPU**, and there is **no `--mem` that buys more** — the second GPU is how you buy the second 120 GB. ⭐ **D5 itself is not the cost** (`+0.02 GB`); `dyneval` is.
> ⚠ **A D5-only re-resume is different and much cheaper** — `dyneval` is lifted from the journal, never re-run, so that pass peaks at a few GB and fits `-G 1 -c 12`. ⛔ **Run 3 is not that**: it is a fresh training leg in a fresh `$RUN3_OUT` and it *will* execute `dyneval`.

> ## ⟲ v2 — WHAT CHANGED AND WHY (read this if you saw v1)
> ⛔ **v1's `sbatch` lines carried `ttt_normalized_write=true` inside `MEM=`. That was WRONG two ways** and is **deleted** — found by `c3-gb-landing` measuring it against the real banked journal, not by reading it. `ttt_normalized_write` is a **`PilotConfig`** field, not a `StreamMemoryConfig` one, so via `--mem` it was **silently dropped** (the TTT fix would not have been on) *and* it still landed in `pilot.memory` as **a second differing key**, which the exemption **refuses**. Routing it through `SET=` is refused too (`pilot.ttt_normalized_write`).
> ⭐ **HUB RULING (option A) — the flag does not travel on run 3.** This is a **scope clarification of PILOT-TTT-RULINGS ruling 1, not an override**: run 3 is the **leak ablation**, and a second token invalidates its attribution **regardless of the exemption** (`PREREG-LeakAblation` §4). Ruling 1's flag belongs to the TTT arm's **own** leg — the **C3 ladder**, which is fresh G-B geometry, needs no exemption, and already sets it (`PREREG-C3-LADDER` §6.4). ⭐ Keeping the TTT arm **bit-identical to run 2, NaN and all, is exactly what a one-token ablation requires** — run 3 is not measuring the TTT arm. **Nothing run 3 exists for is lost.** ⚠ Advisor to confirm the scope reading.

> **How to use this file.** Every fenced block is copy-able **as written**. Anything you must supply is in `<ANGLE BRACKETS>` and appears in **§0**, once. ⛔ Do not "tidy" a command into a shell variable — **zsh does not word-split**, and a variable-built argument list submits garbage (this has bitten this program live).

---

## ✅ BOTH BLOCKERS CLEARED (2026-08-13)

**B2 ✅ CLEARED.** Run 3 must use **run 2's** script, `scripts/csf3/job_gpu_cluformer.sh`, and it had no `PREREG_CONT` passthrough (the exemption spoke had added it only to `job_gpu_c3_seeds.sh`). ⛔ Re-routing through the ladder script was never available: it narrows `--arms`, and `arms` is a `PilotConfig` field ⇒ a second differing key ⇒ **the exemption refuses it**. The two-line passthrough landed in `c3-gb-landing @ e0a212f`; `grep -c 'prereg-continuation' scripts/csf3/job_gpu_cluformer.sh` now returns **1**, asserted by test plus an emitted-command-line/word-splitting test.

**B1 ✅ CLEARED by the Hub merge** — `c3-gb-landing` extends the exemption branch, so **one merge landed both**. §1's tip check below is what you verify it with.

⭐ **`c3-gb-landing` re-verified §1–§6 line by line against the merged code**: the `.eqx` names match `ckpt_path(out, arm, seed)` exactly for all five arms; the `_S4.json`/`_PARTIAL.json` names match for seeds 0–2; and **§3's re-derivation command was executed against the banked seed-0 journal and reproduces §3's `MEM`/`STORE`/`SET` block byte-for-byte**. The only defect found was v1's `ttt_normalized_write`, now removed.

---

## ⛔⛔ §0.0 — PRE-FLIGHT: TWO THINGS TO KNOW BEFORE YOU TOUCH CSF3

### (a) ⚠ THE ARM THAT IS MISSING FROM RUNS 1–2 IS **`ttt_matched`**, NOT A SEED

**Head, 2026-08-13, direct knowledge of the machine: all three seeds are present on CSF3; the TTT arm is not.** This matches the standing record (`c3-handover.md`: *"runs 1–2 DONE (no TTT arm; = the pre-registered ψ ablation)"*) and the pilot-ttt investigation, which measured `MatchedTTTCell.write` diverging to NaN at pilot geometry.

⛔ **A local pulled copy of `csf3_outs/` is NOT a mirror of CSF3** — it is whatever was rsync'd, and it under-reports. **Judge run-2 completeness on CSF3, not from the local tree.**

**What this means for run 3, and it is all good news:**
- **`arms` is at its DEFAULT in run 2's journal** (its flag block carries no `arms` key — verified). So run 3 submitting the default five arms is **config-identical**; it is **not** a second differing key and the exemption is unaffected. ⛔ **Do NOT set `ARMS=` to exclude the TTT arm** — `arms` IS a `PilotConfig` field, so narrowing it would become the second token and the exemption would refuse the leg.
- ⚠ **Run 3's `ttt_matched` arm will train and NaN**, exactly as in run 2, because `ttt_normalized_write` does not travel on this run (v2 note, option A). **That is the one-token rule working, and it is symmetric with run 2** — the leak ablation is unaffected. Its compute is the price of the one-token rule; do not "save" it by narrowing `ARMS`.

### (b) ⛔ BACK UP RUN 2's OUTPUTS BEFORE ANY GIT OPERATION

`.claude/**` is gitignored, so run 2's journals and `.eqx` files are **untracked and not in any remote** — they exist **only** on CSF3. A normal `git checkout`/`pull` leaves untracked files alone, **but if `.claude` was ever force-added on that machine, a checkout will delete it** (this has wiped a worktree in this program before). These journals are the **irreplaceable** input to run 3's exemption and represent ~4 × 16 h of A100 training.

```bash
cp -a ~/scratch/CHLU/.claude/outputs/cluformer-pilot ~/scratch/run2_backup_$(date +%Y%m%d)
```

## §0 — the four values you supply, once

⛔ **Do not assume `RUN2_OUT` — find it.** The journals must be the *live CSF3* ones, not a pulled copy:

```bash
find ~/scratch -name 'pilot_pilot_seed0_PARTIAL.json' -not -path '*/run2_backup*' 2>/dev/null
```

Set `RUN2_OUT` to the **directory that command prints** (without the filename), then:

```bash
export RUN2_OUT=<the directory the find printed>
export RUN3_OUT=~/scratch/CHLU/.claude/outputs/cluformer-run3      # run 3 writes here — must NOT be run 2's dir
export CLU_MAIL=<your-email>
export PREREG_MD=.claude/outputs/c2w6-anti-erosion/PREREG-LeakAblation.md
```

⚠ **These digests are from the LOCAL pulled copy and are NOT authoritative for CSF3** (the Head confirmed the local `csf3_outs/` tree under-reports the machine). ⇒ **the `sha256=` pin is OMITTED from the submit lines** so a stale local digest cannot cause a spurious refusal. To pin anyway — recommended once you are on the machine — compute them THERE and append `sha256=<digest>` inside that seed's `PREREG_CONT`:

```bash
for S in 0 1 2; do shasum -a 256 $RUN2_OUT/pilot_pilot_seed${S}_PARTIAL.json; done
```
| seed | expected sha256 |
|---|---|
| 0 | `43e6598ddcdf9ecdd8b2fd9aec089905c9c2bd85eb44bc1d7db1c3d088dcc2b7` |
| 1 | `62881f3ed6bfcf11c459d4124c13aa2c6c0e4f5a90ec5f7003a46f8c9f417811` |
| 2 | `dd803b279a46a9a8c551b071f06d14ed5bea783852b090afe0d69770d05457eb` |

⚠ **A mismatch is not a reason to drop the `sha256=` pin.** It means CSF3's journal differs from the one verified here — find out **which is authoritative** first. The pin exists to catch exactly this.

## §1 — pull the code on CSF3 (FIRST — `clu-dev`, never `origin`)

```bash
cd ~/scratch/CHLU && git fetch clu-dev && git checkout main && git pull clu-dev main && git log --oneline -1
```
Expected: the tip is **the exemption merge** (B1 cleared). ⛔ If it still prints `0644c48 [hub] merge c3-csf3-harness`, **stop — B1 has not cleared.**

```bash
grep -c 'prereg-continuation' scripts/csf3/job_gpu_cluformer.sh
```
Expected: **`1`**. ⛔ If it prints `0`, **stop — B2 has not cleared.**

## §2 — the `.eqx` inventory (⚠ SCOPE CORRECTED — informational for run 3, blocking for a D5 re-resume)

⚠ **Read the scope before you act on the output.** PILOT-TTT-RULINGS **ruling 2** makes this check **blocking before any D5 *re-resume* of a finished leg**, because a missing checkpoint there costs 16 h silently. ⭐ **Run 3 is not a re-resume** — it is a **fresh training leg in a fresh `$RUN3_OUT`**, and it reads run 2's journal **only as the exemption's comparison reference**, never as a resume source. ⇒ For run 3 this inventory is **informational**: it records what run 2 actually left behind.

Back up `S4.json` first (harmless if a seed has none):
```bash
cd ~/scratch/CHLU && for S in 0 1 2; do [ -f "$RUN2_OUT/pilot_pilot_seed${S}_S4.json" ] && cp -v $RUN2_OUT/pilot_pilot_seed${S}_S4.json $RUN2_OUT/pilot_pilot_seed${S}_S4.json.bak || echo "no S4 for seed$S"; done
```

```bash
cd ~/scratch/CHLU && for S in 0 1 2; do for A in clu_store gru_matched ttt_matched none echo; do [ -f "$RUN2_OUT/ckpt_${A}_seed${S}.eqx" ] && echo "OK      seed$S $A" || echo "MISSING seed$S $A"; done; done
```
**Expected: `ttt_matched` MISSING on every seed** (that arm is absent from runs 1–2) and the other four **OK on all three seeds**. ⛔ **A missing `ttt_matched` is NOT a reason to stop** — run 3 trains that arm from scratch. ⛔ **Any of the other four missing IS worth stopping for**: it means run 2 is less complete than believed, and you should say so before spending four days.

## §3 — regenerate run 2's exact flags (⛔ do NOT transcribe them by hand)

The exemption verifies **bit-identity** with run 2 except the one registered flag, so a single typo fails the run. Print them from the journal itself:

```bash
cd ~/scratch/CHLU && python3 -c "
import json,sys
p=json.load(open(sys.argv[1]))['flags']['pilot']
kv=lambda x:' '.join(f'{k}='+(json.dumps(v) if isinstance(v,bool) else str(v)) for k,v in x.items())
r={k:v for k,v in p.items() if k not in ('memory','store')}
print('MEM=\"%s\"'   % kv(p.get('memory',{})))
print('STORE=\"%s\"' % kv(p.get('store',{})))
print('SET=\"%s\"'   % kv(r))
" $RUN2_OUT/pilot_pilot_seed0_PARTIAL.json
```

Verified against the banked seed-0 journal, this prints exactly:

```
MEM="chunk=64 address_steps=64 read_steps=64 traj_stride=8 psi_hidden=128 write_inner_steps=40 write_n_perturb=8 retry_rounds=1 conv_kernel=4 mlp_mult=4 atom_place_radius=0.3 remat_chunks=true psi_payload_residual=true psi_residual_source=q_star"
STORE="write_margin=0.6"
SET="steps=4000 warmup=200 eval_batches=40 dyneval_batches=40 monitor_every=25 plan_workers=8 liveness_lanes=1"
```

⚠ **`erosion_partition=true` is appended to `MEM` at submission (§4) — it is the ONE registered flag and must not be in the block above.**

## §4 — SUBMIT: one seed per job, `-G 2 -c 24 -t 4-00:00:00`

⛔⛔ **`-G 2 -c 24` IS LOAD-BEARING, NOT A DEFAULT** — see the v3 note. 240 GB against a measured 219.08 GB peak. Dropping to the script's `-G 1 -c 8` (80 GB) OOM-kills the leg hours in, as it already did four times.

⭐ **`PREREG_CONT` must be on EVERY submission line.** Without it the budget refuses the leg.
⭐ Each line is **one literal command**. ⛔ **`ttt_normalized_write` is NOT set here** — see the v2 note above; it travels on the C3 ladder's TTT leg, not on this ablation. ⚠ Expect the run-3 TTT arm to NaN exactly as it did in run 2; **that is the one-token requirement working, not a regression.**

```bash
cd ~/scratch/CHLU && mkdir -p logs
```

```bash
sbatch --export=ALL,SEEDS="0",STAGE=pilot,STG=s4,D5=1,RESUME=1,OUT="$RUN3_OUT",MEM="chunk=64 address_steps=64 read_steps=64 traj_stride=8 psi_hidden=128 write_inner_steps=40 write_n_perturb=8 retry_rounds=1 conv_kernel=4 mlp_mult=4 atom_place_radius=0.3 remat_chunks=true psi_payload_residual=true psi_residual_source=q_star erosion_partition=true",STORE="write_margin=0.6",SET="steps=4000 warmup=200 eval_batches=40 dyneval_batches=40 monitor_every=25 plan_workers=8 liveness_lanes=1",PREREG_CONT="journal=$RUN2_OUT/pilot_pilot_seed0_PARTIAL.json flag=memory.erosion_partition prereg=$PREREG_MD" --mail-user=$CLU_MAIL -G 2 -c 24 -t 4-00:00:00 --job-name=clu-run3-s0 scripts/csf3/job_gpu_cluformer.sh
```

```bash
sbatch --export=ALL,SEEDS="1",STAGE=pilot,STG=s4,D5=1,RESUME=1,OUT="$RUN3_OUT",MEM="chunk=64 address_steps=64 read_steps=64 traj_stride=8 psi_hidden=128 write_inner_steps=40 write_n_perturb=8 retry_rounds=1 conv_kernel=4 mlp_mult=4 atom_place_radius=0.3 remat_chunks=true psi_payload_residual=true psi_residual_source=q_star erosion_partition=true",STORE="write_margin=0.6",SET="steps=4000 warmup=200 eval_batches=40 dyneval_batches=40 monitor_every=25 plan_workers=8 liveness_lanes=1",PREREG_CONT="journal=$RUN2_OUT/pilot_pilot_seed1_PARTIAL.json flag=memory.erosion_partition prereg=$PREREG_MD" --mail-user=$CLU_MAIL -G 2 -c 24 -t 4-00:00:00 --job-name=clu-run3-s1 scripts/csf3/job_gpu_cluformer.sh
```

```bash
sbatch --export=ALL,SEEDS="2",STAGE=pilot,STG=s4,D5=1,RESUME=1,OUT="$RUN3_OUT",MEM="chunk=64 address_steps=64 read_steps=64 traj_stride=8 psi_hidden=128 write_inner_steps=40 write_n_perturb=8 retry_rounds=1 conv_kernel=4 mlp_mult=4 atom_place_radius=0.3 remat_chunks=true psi_payload_residual=true psi_residual_source=q_star erosion_partition=true",STORE="write_margin=0.6",SET="steps=4000 warmup=200 eval_batches=40 dyneval_batches=40 monitor_every=25 plan_workers=8 liveness_lanes=1",PREREG_CONT="journal=$RUN2_OUT/pilot_pilot_seed2_PARTIAL.json flag=memory.erosion_partition prereg=$PREREG_MD" --mail-user=$CLU_MAIL -G 2 -c 24 -t 4-00:00:00 --job-name=clu-run3-s2 scripts/csf3/job_gpu_cluformer.sh
```

⚠ The `sha256=` pin is included for **seed 0** (the digest verified by the exemption spoke). Pin seeds 1–2 too if you have their digests — it is optional and recommended, and a mismatch is refused.

## §5 — verify the jobs exist BEFORE walking away

```bash
squeue -u $USER --name=clu-run3-s0,clu-run3-s1,clu-run3-s2; ls -l logs/clu-run3-*.out 2>/dev/null | wc -l
```
Expected: **3 jobs queued/running**. ⚠ A launch that produced no log is a launch that did not happen — do not assume.

Confirm the exemption was actually taken (not silently skipped):
```bash
grep -h 'preregistered_continuation\|budget_exempted' $RUN3_OUT/*.json | head
```
Expected: `budget_exempted: true` with the journal path and sha256 stamped.

## §6 — ⛔ EVERY RE-RESUME CARRIES `PREREG_CONT` TOO

**Stated twice because the leg is refused without it.** A re-resume is the *same* line from §4 (same seed, same `PREREG_CONT`, `RESUME=1`) — re-submit it verbatim. ⛔ Do not shorten it "because it is only a resume."

## §7 — pull the artifacts

```bash
rsync -av csf3:~/scratch/CHLU/.claude/outputs/cluformer-run3/ ./.claude/outputs/cluformer-run3/
```

---

### Provenance
`main @ 0644c48` · run-2 journals `.claude/outputs/cluformer-pilot/csf3_outs/run2/` (seeds 0–2, `_PARTIAL` + `_S4`) · flags re-derived from `pilot_pilot_seed0_PARTIAL.json` by the §3 command, not transcribed · exemption spec `.claude/outputs/c3-run3-budget-exemption.md` §6 · rulings `.claude/outputs/c2w11/PILOT-TTT-RULINGS.md` 1–3 · `PREREG-LeakAblation.md` §4 (one-token rule). ⛔ `origin` frozen at `40c2f31`.
