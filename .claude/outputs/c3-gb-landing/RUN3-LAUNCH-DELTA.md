# RUN3-LAUNCH-DELTA — what `c3-gb-landing` verified, and the ONE line that must change

**Filed 2026-08-13 by the `c3-gb-landing` spoke.** ⛔ **I did not edit `RUN3-LAUNCH.md` — it is the Hub's
document.** This is the delta, in paste-able form.

---

## ✅ B2 IS CLEARED

```
$ grep -c 'prereg-continuation' scripts/csf3/job_gpu_cluformer.sh
1
$ bash -n scripts/csf3/job_gpu_cluformer.sh      # clean
```
On branch `agent/experiment-engineer/c3-gb-landing` @ `e0a212f`. RUN3-LAUNCH §1's own gate now passes, and
it is asserted by `tests/test_c3_gb_geometry.py::test_run2s_job_script_carries_the_PREREG_CONT_passthrough`
plus an emitted-command-line/word-splitting test.
⚠ **B1 is unchanged and still open**: the exemption branch (now this branch, which extends it) is still
unmerged, so a CSF3 `git pull` of `main` would still fetch code without `--prereg-continuation`.

## §1–§4, line by line, against this branch

| line | verdict |
|---|---|
| §1 `git fetch clu-dev && git checkout main && git pull clu-dev main` | ✅ works; the tip check is a Hub/Head matter (B1) |
| §1 `grep -c 'prereg-continuation' … job_gpu_cluformer.sh` → **1** | ✅ **now 1** (was 0 — this was B2) |
| §2 `cp $RUN2_OUT/pilot_pilot_seed${S}_S4.json …bak` | ✅ names match `partial_path`/the S4 artifact; both `_PARTIAL.json` and `_S4.json` exist for seeds 0–2 in the banked run-2 dir |
| §2 `[ -f "$RUN2_OUT/ckpt_${A}_seed${S}.eqx" ]` | ✅ matches `ckpt_path(out, arm, seed)` = `ckpt_{arm}_seed{seed}.eqx` exactly, for all five arm names |
| §3 the `python3 -c` flag re-derivation | ✅ **executed against the banked seed-0 journal; it reproduces the document's MEM/STORE/SET block byte-for-byte** |
| §4 the three `sbatch` lines | ⛔ **DEFECT — see below.** Everything else in them is consumed by the script: `SEEDS STAGE STG D5 RESUME OUT MEM STORE SET PREREG_CONT` ✅, no `ARMS` ✅ (so `arms` stays default), `sha256=` is an accepted optional spec key ✅ |
| §5 `grep -h 'preregistered_continuation\|budget_exempted'` | ✅ both keys exist in the emitted artifact |
| §6 "every re-resume carries `PREREG_CONT`" | ✅ true, and now possible on this script |

---

## ⛔⛔ THE DEFECT IN §4 — `ttt_normalized_write=true` INSIDE `MEM=` BREAKS RUN 3 TWO WAYS

`ttt_normalized_write` is a **`PilotConfig`** field. It is **not** a `StreamMemoryConfig` field. Measured
against the **real** banked journal `csf3_outs/run2/pilot_pilot_seed0_PARTIAL.json`:

```
pcfg.ttt_normalized_write = False        <- passed via --mem: SILENTLY DROPPED
  (StreamMemoryConfig.from_mapping filters unknown keys)
raw pilot.memory carries it: True        <- but the OVERRIDE DICT keeps it

exemption, run2 + erosion_partition only        : ACCEPTED  ['memory.erosion_partition', 'pilot.memory']
exemption, RUN3-LAUNCH §4 literal (MEM=…ttt_nw) : REFUSED
    pilot.memory: journal={…} now={… "ttt_normalized_write": true …}
exemption, SET="… ttt_normalized_write=true"    : REFUSED
    pilot.ttt_normalized_write: journal=<absent> now=true
```

⇒ **as written, §4 submits a leg that (a) does not have the TTT fix on, and (b) is refused before training
with `StateByteBudgetError`/`ContinuationExemptionError`.** ⛔ And the second route is refused too: **there
is no way to set `ttt_normalized_write` on run 3 and keep the exemption**, because the exemption admits
**exactly one** registered flag and `PREREG-LeakAblation` §4 admits exactly one token.

⚠ **This is not a plumbing bug and it is not fixable by plumbing.** It is a real conflict between
PILOT-TTT-RULINGS **ruling 1** (the TTT arm is submitted with the flag set) and PREREG-LeakAblation **§4**
(run 3 changes exactly one token). ⛔ **It is the Hub's/Advisor's to rule, not mine** — the exemption's
verification logic is explicitly outside my ownership and I did not touch it.

### The three ways out, costed

| option | what it costs | comment |
|---|---|---|
| ⭐ **A — drop `ttt_normalized_write=true` from run 3** (run 3 = run 2 + `erosion_partition`, exactly) | the run-3 TTT arm NaNs again, as in run 2 | ⭐ **Recommended.** Run 3's purpose is the **leak ablation**, whose attribution the second token would invalidate *regardless of the exemption*. Ruling 1's flag belongs to the arm's *own* leg — and the **C3 ladder** is that leg: it is a fresh geometry (G-B), needs no exemption, and `PREREG-C3-LADDER` §6.4 sets the flag there. **Nothing is lost that run 3 was for.** |
| B — set the flag and drop the exemption (`SET="… enforce_state_byte_budget=false ttt_normalized_write=true"`, no `PREREG_CONT`) | the artifact records `enforced: false`; ⛔ the leak-ablation attribution is still broken by the second token | trades a stamped, verified exemption for an unenforced budget **and does not fix the attribution**. |
| C — register a second flag | ⛔ a new pre-registration **and** a code change, by design | the exemption refuses lists/globs by construction; this is what that refusal is for. |

### If option A is taken, §4's three lines change by exactly one deletion

Delete ` ttt_normalized_write=true` from the `MEM=` string in each of the three `sbatch` lines. Nothing
else moves. Seed 0's line then reads:

```bash
sbatch --export=ALL,SEEDS="0",STAGE=pilot,STG=s4,D5=1,RESUME=1,OUT="$RUN3_OUT",MEM="chunk=64 address_steps=64 read_steps=64 traj_stride=8 psi_hidden=128 write_inner_steps=40 write_n_perturb=8 retry_rounds=1 conv_kernel=4 mlp_mult=4 atom_place_radius=0.3 remat_chunks=true psi_payload_residual=true psi_residual_source=q_star erosion_partition=true",STORE="write_margin=0.6",SET="steps=4000 warmup=200 eval_batches=40 dyneval_batches=40 monitor_every=25 plan_workers=8 liveness_lanes=1",PREREG_CONT="journal=$RUN2_OUT/pilot_pilot_seed0_PARTIAL.json flag=memory.erosion_partition prereg=$PREREG_MD sha256=43e6598ddcdf9ecdd8b2fd9aec089905c9c2bd85eb44bc1d7db1c3d088dcc2b7" --mail-user=$CLU_MAIL -t 4-00:00:00 --job-name=clu-run3-s0 scripts/csf3/job_gpu_cluformer.sh
```

⭐ **Verified**: exactly this configuration (run 2's flags + `erosion_partition=true`, nothing else) is
**ACCEPTED** by `verify_preregistered_continuation` against the real seed-0 journal, moving exactly
`['memory.erosion_partition', 'pilot.memory']`.

### Recommended pre-submit assertion, to be added to §5 (one line, catches the whole class)

```bash
grep -o '"ttt_normalized_write":[^,}]*' $RUN3_OUT/pilot_pilot_seed0_PARTIAL.json | head -1
```
Expected: **nothing** under option A. ⛔ Any output means a second token is in the leg and the ablation's
attribution is void.

---

### Provenance
Branch `agent/experiment-engineer/c3-gb-landing` @ `7d33308` (off `c3-run3-budget-exemption` @ `a656746`,
off `main` @ `0644c48`) · probe `.claude/scratch/c3-gb-landing/probe_run3_launch.py`, run against the real
`csf3_outs/run2/pilot_pilot_seed0_PARTIAL.json` · JAX 0.9.0, main venv reused via `PYTHONPATH`, CPU/float32
· the behaviour is pinned by `tests/test_c3_gb_geometry.py::
test_ttt_normalized_write_is_NOT_a_memory_field_and_MEM_would_DROP_it`.
