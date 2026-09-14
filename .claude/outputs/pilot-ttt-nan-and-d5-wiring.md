# pilot-ttt-nan-and-d5-wiring — experiment-engineer report

**Task + acceptance criterion:** name+fix the `ttt_matched` NaN mechanism (or stop-and-report if claims-relevant), wire D5 through the job script with its host-memory cost projected against the 251 GB envelope, and answer *mechanically* whether D5 can be obtained by re-resuming a finished leg — both gates green, resume compatibility demonstrated.

**Status: done**, with **one deliberate STOP** (the TTT fix is built, measured and gated OFF; flipping the default is a Hub ruling) and **one unrequested blocking defect found and fixed** (see §0 — it invalidated the banked training *before* I touched anything).

**Dial declaration.** Dial: **none — instrument/defect repair.** No performance claim is made here. Laundering control: n/a (the only numbers are stability diagnostics and RSS projections). Falsifies: the TTT arm NaN-ing again at pilot shape with the lever ON, or a banked journal being refused. Does NOT falsify: the TTT arm *beating* the CLU once it stops diverging — that is a rival column doing its job, not a defect in this work.

> ⚠ **RECONCILIATION LIST — needs an owner (protocol §5).** Three items in §7 below require a Hub decision or a downstream edit: (1) the **ruling on `ttt_normalized_write`'s default**, (2) the **`.eqx` precondition check on CSF3** before any D5 re-resume, (3) the standing statement *"the toy bit-identity gate protects the scale run"* — it is **false for this arm** and the reason is now measured.

---

## 0. ⛔⛔ FIRST — the banked training was ALREADY unspendable on `main`

Before my change, on `main @ 80d7d4b`, **`load_journal` refuses the real landed CSF3 journal**. Verified directly against `.claude/outputs/cluformer-pilot/csf3_outs/pilot_pilot_seed0_PARTIAL.json`:

```
differing fingerprint keys: ['memory.erosion_partition', 'memory.refresh_amp_ceiling',
                             'memory.refresh_max_gain', 'memory.refresh_monotonic']
⛔ refusing to resume: … was written under a DIFFERENT config.
    memory.erosion_partition: journal=<absent> now=false
    memory.refresh_amp_ceiling: journal=<absent> now=0.0
    memory.refresh_max_gain: journal=<absent> now=4.0
    memory.refresh_monotonic: journal=<absent> now=false
```

**Cause (structural, not a typo).** `_flag_dict` builds the `pilot` and `store` groups from `as_flag_table()` — **non-default keys only**, so a new field is invisible — but the `memory` group from a full `asdict(StreamMemoryConfig())`. So **every field ever added to `StreamMemoryConfig` retro-invalidates every journal on disk.** The four offenders arrived in C2W6 at `ffe7440` ("P1 stop-gradient partition + I1 refresh-on-rewrite guard"), long after the legs launched.

**Consequence for the task as scoped:** the "~free D5 re-resume" was *already* impossible the moment CSF3 pulls a current checkout, and 4 × ~16 h of banked A100 training was already stranded — by plumbing, exactly like Defect 2. The hard constraint could not be met by *avoiding* the fingerprint; it had to be repaired.

**Fix (`171972d`).** A key the journal **predates**, sitting at its **own field default**, is the same leg — because this repo ships every new lever gated OFF and bit-identical. Those keys are reconciled and printed; everything else stays strict:

| case | before | after |
|---|---|---|
| journal predates a memory field, current value = field default | ⛔ refuse | ✅ accept (+ printed) |
| journal predates a memory field, current value ≠ default | ⛔ refuse | ⛔ refuse |
| journal has a key the code no longer has | ⛔ refuse | ⛔ refuse |
| shared key, different value (`write_inner_steps=3`) | ⛔ refuse | ⛔ refuse |

After the fix, on the real journal: `✅ load_journal ACCEPTED the real CSF3 journal; 5 arms lifted`.

⚠ **This is a loosening of a §A20.4 provenance guard and the Hub should review it as such.** It only ever accepts when the current value is provably the field default; the soundness premise is the *existing* repo convention (new levers ship OFF + bit-identical), which is asserted by each wave's own gate.

---

## 1. DEFECT 1 — the TTT NaN: mechanism, evidence, and the STOP

### 1.1 The mechanism (named, measured)

`MatchedTTTCell.write` is a raw gradient step with a **global** `eta`:

```
W ← W (I − η k kᵀ) + η v kᵀ ,   k = θ_K z
```

The factor along `k` is `1 − η‖k‖²`, so the chunk recursion is non-expansive **only while `η‖k‖² < 2`**. But

```
‖k‖² = ‖θ_K z‖² ≈ n ‖z‖² / d        (θ_K is (n,d) at scale 1/√d)
```

so the stability product is `η·n/d` — **a pure function of the solved geometry**, and `n` comes from `solve_matched_ttt`, which reads it off **the CLU cell's byte ledger**.

> ⛔ **The two-sided byte match — the thing that makes the swap fair — silently chooses the rival's inner-loop stability, and nothing checks it.**

| | ledger `(P,S,d)` | solved `(k,n)` | `η·n/d` | measured on **real φ latents** | ‖W‖ over 16 chunks (one forward) |
|---|---|---|---|---|---|
| **pilot** | (168986, 115072, **12**) | **(2197, 52)** | **3.00** | **3.47 mean, >2 on 100 % of chunks** | 18.9 → **1.49e6** (×7.9e4) |
| toy | (8616, 5144, **3**) | (571, 9) | 2.08 | 2.31 mean, >2 on **44 %** of chunks | 3.16 → 23.1 (**bounded**) |

At pilot every chunk amplifies, so the growth is monotone over the 16 chunks of `seq_len/chunk = 1024/64`; at toy fewer than half do, so it is intermittent and self-limiting. **The toy is marginal, not safe** — and that, not luck, is why every toy gate passed. (Read magnitude injected into the residual stream at the last chunk: **1.7e4** at pilot vs **0.17** at toy.)

Training then closes the loop: `optax.clip_by_global_norm` cannot help because the **forward** overflows, and a NaN global-norm propagates through the clip.

### 1.2 Reproduced in 33 seconds — and this is the answer to task §1.3

⚠ Task §1.3 asks what evidence would convince us a fix holds at pilot shape without burning another 22 h leg. **The answer is that pilot *shape* is not what matters — pilot *memory geometry* is**, because the criterion `η·n/d` depends only on `(addr_dim, payload_dim, capacity, atoms_per_item)` and not on `d_model`, `n_layers` or `batch`. So the NaN reproduces on a laptop:

**Config: pilot memory geometry + pilot LR schedule (`lr 1e-3`, `warmup 200`, `steps 4000`), toy-sized shell (`d_model=64, n_layers=2, batch=2`), seed 0.**

| run | first non-finite loss step | best loss reached | ‖W‖ growth in one forward |
|---|---|---|---|
| **shipped (`ttt_normalized_write=False`)** | **107** / 300 | 4.754 (then NaN) | ×6.26e4 |
| **fixed (`ttt_normalized_write=True`)** | **NONE** in 1000 steps | **2.117** | ×1.63 |
| toy memory geometry, shipped | **NONE** in 300 steps | 2.585 | — |

CSF3 saw NaN at step **135**/4000 on the same LR schedule; the cheap rig sees it at **107**. Same failure, same schedule position (inside warmup), **1/2400th of the cost**.

⇒ **The proposed gate for any TTT rerun: 300 steps at pilot memory geometry with a toy shell, on the pilot LR schedule. It is ~33 s, it is positive-control-verified (it *does* NaN on the shipped code), and it is scale-sufficient because the criterion is geometry-only.**

### 1.3 The fix, and the ⛔ STOP

`ttt_normalized_write=True` switches the cell to the **normalized delta rule / Kaczmarz step**:

```
W ← W − η (W k − v) kᵀ / (‖k‖² + 1e-6)
```

Why this and not "lower the LR":

* it is **what the cell's own docstring already claims** — *"one **closed-form** step"*. A closed-form solve of `min_W ‖Wk − v‖²` along `k` is exactly `η = 1` here (asserted in a test: `W₂k == v`); the shipped code does a *gradient* step instead.
* it is **TTT-Linear's own `1/d` factor** on the token-wise inner learning rate (Sun et al. 2024), which our port dropped.
* it is **non-expansive for every `η ∈ (0,2)` and scale-free in `n`, `d`, `‖z‖`** — measured: growth over 16 coherent writes is **1.443 at the pilot ledger and 1.443 at the toy ledger**, identical to 1e-3. **A future ledger solve cannot re-break it**, which a re-tuned constant would not guarantee.
* it costs **zero parameters and zero state bytes** (`eqx.field(static=True)`; `cell_ledger()` asserted identical), so the published swap ledger does not move.

⛔ **I have NOT made it the default, and I am stopping here for the Hub.** It is a claims-relevant change to a published rival column, and not only because of the NaN: at 1000 steps the fixed arm reaches **2.12** where the shipped arm's best-before-NaN was **4.75**. **The fix makes the rival substantially STRONGER**, which is the honest direction but is precisely a Hub call.

**Proposed value: `ttt_normalized_write=True` for all six tier-iii legs.** The three options as I see them:

| option | consequence |
|---|---|
| **(a) flip the default** | every artifact, incl. banked journals, becomes a different leg (see §3) — a full 6-leg rerun of the TTT arm |
| **(b) leave default OFF, submit the TTT arm with `SET="ttt_normalized_write=true"`** | honest flag in the artifact; the flag is non-default so the leg is *declared* different; the other four arms' banked results are untouched but must be re-resumed **in a separate `--out`** |
| **(c) publish the NaN** | "the matched TTT rival diverges by construction at the matched geometry" is a real finding about the byte match, but it is not a rival column and DF1/DF3 cannot be adjudicated without one |

I recommend **(b)** — it keeps every landed number intact and adds the rival column as a declared, separately-flagged leg.

---

## 2. DEFECT 2 — D5 wiring, cost, and the ⭐ re-resume answer

### 2.1 Wiring (`2469ba5`)

```sh
D5="${D5:-0}"                                    # + a documented header block
…
[ "$D5" = "1" ] && EXTRA="$EXTRA --d5"           # the RESUME idiom, name-for-name
echo "=== config overrides === … RESUME='$RESUME' D5='$D5'"
```

`bash -n` clean; verified with `set -eo pipefail` that the false branch does **not** trip `set -e` (it is an AND-list — same as `RESUME`). Verified by execution: `D5=1 RESUME=1 → EXTRA=[ --resume --d5]`; `D5` unset → `EXTRA=[]`, exit 0. A test asserts the D5 line equals the RESUME line with the names substituted, plus a passthrough-completeness test over all seven flags (the N-registry item: *a pre-registered phase gated behind a flag no launch path sets is indistinguishable from a deliberate cut*).

### 2.2 ⭐ **CAN D5 BE OBTAINED BY RE-RESUMING A FINISHED LEG? — YES. It is ~free.**

Three independent confirmations:

**(i) The mechanism.** `--d5` is a **CLI argument, not a `PilotConfig` field**. It never enters `rec["flags"]`, so it is invisible to `_flag_dict` and **cannot move the resume fingerprint** (asserted in a test). Inside `run_pilot`, `_Phases.step(key, …)` lifts any key already in the journal; `anytime_curve` is the one key a pre-D5 journal does not have, so it is the one phase that executes. `dyneval` — the phase that demanded **219 GB** and killed four legs — is **lifted, never run**.

**(ii) Replayed against the REAL landed journal** (`d5_dryrun.py`, decisions taken straight from `run_pilot`):

```
WOULD RUN  :  clu_store/anytime_curve
WOULD LIFT :  22 phases — phi_gain_calibrated, monitors_init, allocation_liveness_init,
              gradient_probe_init, clu_store/{train,static,dyneval,blank_store,
              gradient_probe_final,selectors_final}, {gru_matched,ttt_matched,none,echo}/
              {train,static,dyneval}
```

**1 phase runs, 22 lifted. No training. No dyneval.**

**(iii) Executed end-to-end at toy** (`test_re_resuming_a_FINISHED_leg_with_d5_runs_only_the_anytime_curve`): a *finished* leg re-resumed with `d5=True` logs `training SKIPPED` for every arm, `lifted from the journal` for every eval phase, `[rss] clu_store/anytime_curve/enter` for exactly one, and the resulting artifact is **bit-identical to the `d5=False` one with the `anytime_curve` key removed** (0 differing leaves).

⛔ **Four preconditions, all mechanical:**
1. `ckpt_{arm}_seed<N>.eqx` must be present in `$OUT` **for all five arms**. `run_pilot` gates on `banked is not None and ck.exists()`; **a missing `.eqx` silently RETRAINS that arm (~16 h)**. Check `ls $OUT/ckpt_*_seed<N>.eqx | wc -l` == 5 before submitting.
2. CSF3 must be on a checkout carrying **both** the D5 passthrough **and** §0's journal reconciliation. On today's `main` the resume is refused (§0); on the legs' original commit there is no `--d5`. **This branch is the only tree where the re-resume works.**
3. ⛔ **Do NOT pass `ARMS`** on the re-resume. `arms` *is* a `PilotConfig` field, so `ARMS="clu_store"` makes it non-default, changes the fingerprint, and the journal is refused.
4. The final `pilot_pilot_seed<N>_S4.json` is **overwritten** (same numbers + the new key). Back it up first.

### 2.3 Host-memory cost, against the 251 GB envelope

Measured from the landed leg's own `[rss]` series (the `-G 2 -c 24` recovery, seed 0 / run 1):

| phase | rss enter → exit | **Δ** | wall |
|---|---|---|---|
| `clu_store/static` (forward-only, verlet 64+64) | 15.90 → 15.91 | **+0.01 GB** | 340 s |
| `clu_store/blank_store` (forward-only) | 155.99 → 155.94 | **≈ 0** | 539 s |
| `clu_store/gradient_probe_final` | 155.63 → 164.83 | +9.20 GB | 2393 s |
| **`clu_store/dyneval`** | 3.34 → **219.09** | **+215.75 GB** | 61 686 s |
| run peak | | **219.08 GB = 87.3 % of ~251 GB** | |

`anytime_curve` is **five `evaluate()` calls** — structurally *the same call as `static`*, only with a different **static** `verlet`, and `release_host_memory()` between each so they never stack. Pilot budgets are `(8,8) (16,16) (32,32) (64,64) (128,128)` ⇒ verlet-per-read `16, 32, 64, 128, **256**`; the largest is **2×** the trained budget.

**Projection** (assumption stated: an eval program's host footprint is dominated by the traced/compiled unroll, ~linear in `address_steps + read_steps`):

* largest budget ≈ 2 × `static`'s Δ = **≈ 0.02 GB**;
* with hygiene only one is resident ⇒ **phase peak ≈ 0.02 GB**; even the naive no-hygiene sum is `(16+32+64+128+256)/128 × 0.01 =` **0.04 GB**.

**Verdict: D5 fits the `-G 2 -c 24` envelope with ~4 orders of magnitude to spare, and does not move the run peak** (it runs at the ~155.6 GB retained floor, 63 GB *below* the 219.08 GB peak `dyneval` already set). ⭐ **And on the re-resume path `dyneval` is lifted, so the 219 GB spike never happens at all: a D5-only pass peaks at a few GB and fits `-G 1 -c 12` (~120 GB) trivially.**

**Wall time for the D5-only pass:** bounded between 22 min (all cost verlet-scaled: `340 × 496/128`) and 28 min (all cost plan-pass, verlet-independent: `5 × 340`), **plus five one-shot compiles** ⇒ budget **≤ ~1 h/leg** on `-G 1 -c 12`. Versus **~22 h** for a retrain.

---

## 3. Gates

| gate | result |
|---|---|
| **Toy bit-identity** (old `main` vs new, `run_pilot` toy, `D5=0`, TTT fix inert; arms `clu_store, ttt_matched, none`) | ✅ **1073 leaves compared, 0 differing.** Run from a `git worktree` at `main` using the MAIN venv (protocol §4) — JAX 0.9.0 in both. (First pass showed 1 diff, `flags.pilot.data_root`, an artifact of the two harnesses writing their fake corpus to different dirs; re-run with a shared corpus dir ⇒ 0.) |
| **Resume-accept, demonstrated not asserted** | ✅ Against the **real** landed CSF3 journal: refused on `main` (4 keys), `ACCEPTED … 5 arms lifted` on this branch. Plus 5 pytest cases covering accept/refuse in both directions. |
| **`ruff check chlu/ tests/ scripts/`** | ✅ All checks passed. (⚠ `ruff format --check` is **not** a repo gate — it also fails on `main`'s own `blocks.py`; I did not reformat, to keep the diff minimal.) |
| **New tests** | ✅ **18 passed** (`tests/test_ttt_stability_and_d5_wiring.py`, 211 s) |
| **Neighbouring suites** | ✅ **93 passed** (`test_pilot_checkpoint_resume`, `test_cluformer_pilot`, `test_blocks`, `test_csf3_memory_fit`, `test_null_arms`; 636 s) |
| **Full suite** | ✅ **1463 passed / 0 failed** (3148.91 s) — see §6 |

---

## 4. Flag-provenance table

Every number in §1.2 comes from one of two rigs. Commit `7fcef50`, **JAX 0.9.0**, CPU/float32, seed 0 throughout.

| | cheap NaN rig (§1.2) | toy bit-identity gate (§3) |
|---|---|---|
| `scale` | `pilot` | `toy` |
| non-default `PilotConfig` | `d_model=64, n_layers=2, batch=2, steps=4000, warmup=200, eval_batches=1, dyneval_batches=1, data_bytes=4_000_000, store_watch=False, arms=("ttt_matched",)`, `ttt_normalized_write` ∈ {False, True} | `d_model=16, n_layers=2, seq_len=16, batch=2, vocab_size=256, addr_dim=2, payload_dim=1, capacity=4, atoms_per_item=16, steps=2, warmup=1, eval_batches=1, dyneval_batches=1, monitor_every=1, arms=("clu_store","ttt_matched","none")` |
| memory | pilot defaults (`chunk=64, address_steps=64, read_steps=64, traj_stride=8, psi_hidden=128, write_inner_steps=4, write_n_perturb=8, retry_rounds=1, conv_kernel=4, mlp_mult=4`) | `chunk=8, address_steps=4, read_steps=4, traj_stride=2, psi_hidden=8, write_inner_steps=1, write_n_perturb=4, retry_rounds=1, conv_kernel=3, mlp_mult=2` |
| store | pilot defaults + all stage flags ON | `min_atoms=64, min_atoms_base=32` |
| optimiser | `adamw`, peak `lr 1e-3`, warmup-cosine over 4000, `grad_clip=1.0`, `weight_decay=0.0` | same |
| data | real enwik8, first 4 MB | 40 000-byte synthetic stream (`arange % 251`) |
| `with_d5` / `resume` | n/a | `False` / `False` |

The `[rss]` and wall figures in §2.3 are **not mine** — they are read off the landed CSF3 artifact `pilot_pilot_seed0_PARTIAL.json` (run 1, seed 0, `-G 2 -c 24`, `MEM="atom_place_radius=0.3 write_inner_steps=40 remat_chunks=true"`, `STORE="write_margin=0.6"`, `SET="monitor_every=25 plan_workers=8 liveness_lanes=1"`, JAX 0.9.0 / gpu).

**Pre-registration:** the acceptance criterion here is not a measured ratio/exponent/slope/law (it is "reproduce, name, fix, wire"), so `PREREG.md` does not apply. The one predictive claim I did make in advance — *"the NaN is a forward overflow in the TTT inner state, not an optimiser problem, and it will therefore reproduce at pilot **memory geometry** with a toy shell"* — was written before the rig was run and is confirmed (step 107 vs the cluster's 135).

---

## 5. Git footprint

Branch **`agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring`**, off local `main @ 80d7d4b`. Not pushed. Rebase onto `main` is a no-op (base unmoved; `origin/main` deliberately untouched per §7.21).

| commit | files |
|---|---|
| `2469ba5` wire D5 into the CSF3 job script | `scripts/csf3/job_gpu_cluformer.sh` (+20/−2) |
| `1ed0902` name and gate the TTT divergent inner loop | `chlu/core/blocks.py` (+46/−3), `chlu/training/train_cluformer.py` (+25/−2) |
| `171972d` stop a new `StreamMemoryConfig` field invalidating every journal | `chlu/experiments/exp_cluformer_pilot.py` (+34) |
| `7fcef50` tests | `tests/test_ttt_stability_and_d5_wiring.py` (new, 18 tests) |

Worktree `../CHLU-ttt-base` was created at `main` for the bit-identity gate and **removed**; `git worktree list` shows only the main checkout. No shared/unrelated file was touched; `chlu/config.py`, the CLI, `utils/plotting.py` and every other campaign's code are untouched.

Scratch (all under `.claude/scratch/pilot-ttt-nan-and-d5-wiring/`): `probe_ttt.py`, `repro_nan{,2,3}.py`, `kn.py`, `dbg{,2,3,4}.py`, `resume_real.py`, `d5_dryrun.py`, `bitgate{,2}.py`, `bitcmp.py`, `old/`, `new/`, `new2/`.

---

## 6. Full-suite result

```
uv run --no-sync python -m pytest -q -p no:cacheprovider --no-cov
1463 passed, 29 warnings in 3148.91s (0:52:28)
```

**Arithmetic checked, not assumed.** `--collect-only` on a clean worktree at `main @ 80d7d4b` collects **1445**; my branch runs **1463 = 1445 + 18**, i.e. exactly the new file and nothing else. ⚠ The handover's "expect 1363" and the C2W8 review's "1443" are both stale relative to `80d7d4b`, whose true collected count is **1445** — the Hub may want to correct that figure.

**HEAD stability (the standing "a suite run needs a stable HEAD" rule):** HEAD recorded as `7fcef50…` before the run and re-verified `7fcef50…` after; `main` unmoved at `80d7d4b` before and after; working tree clean.

⚠ **Concurrent-agent note:** three other spokes' worktrees (`../CHLU-c2w8a`, `../CHLU-c2w8b`, `../CHLU-c2w8c`) appeared *during* my suite run. They are separate working directories on separate branches, so they cannot have touched this checkout — and HEAD/`main`/tree-clean were verified on both sides of the run, so the result stands. My own temporary worktree (`../CHLU-ttt-base`, for the bit-identity gate and the collect-only count) was created and removed twice and is gone; **my four commits were verified present on the shared ref from the main repo before each removal** (protocol §3.2).

---

## 7. Open questions / follow-ups / risks

1. ⛔ **HUB RULING OWED: `ttt_normalized_write`'s default.** Options (a)/(b)/(c) in §1.3; I recommend (b). Until it is ruled, the tier-iii pilot has **no TTT rival column** and DF1/DF3 cannot be adjudicated.
2. ⛔ **HEAD ACTION OWED before any D5 re-resume:** confirm all five `ckpt_{arm}_seed<N>.eqx` are present in each leg's `$OUT` on CSF3. A missing one costs 16 h silently.
3. ⚠ **The §0 fix is a loosening of a provenance guard** and deserves an explicit Hub sign-off rather than riding along with the D5 work.
4. ⚠ **The reconciliation is one-directional by design.** If a future field is added to `StreamMemoryConfig` whose *default* changes behaviour (violating the ship-OFF convention), a journal would be wrongly accepted. The durable fix is to give `StreamMemoryConfig` an `as_flag_table()` like the other two groups — larger, and it would change the fingerprint of every journal *once*, so I did not do it here.
5. ⚠ **The five in-flight legs are running on an OLDER CSF3 checkout.** Their journals will be resumable by this branch (that is §0), but a leg that is mid-flight when the Head pulls will see the *code* change under it. Recommend: let running legs finish on their current checkout, pull only for the D5 pass.
6. 🔍 **Not investigated (out of scope):** whether `η` (`log_eta`) drifting during training contributes. Under the shipped rule the fixed rig's `η` ends `nan`; under the fix it ends `0.62` from `0.69`, i.e. it barely moves — consistent with the divergence being geometric, not learned.
7. ⚠ **`plan_pass` is 87 % of the TTT arm's training wall** (`plan_pass_frac 0.869` in the landed artifact). Nothing to do with the NaN, but it means a TTT rerun is *not* GPU-bound and the `plan_workers` lever is where its wall time lives.

---

## Proposed handover updates (for the Hub)

**§7 — new entries**

- **7.30 [RESOLVED on `agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring`, needs merge] The `ttt_matched` NaN is the byte match choosing the rival's stability.** `MatchedTTTCell`'s inner update is non-expansive only while `η‖θ_K z‖² < 2`; that product is `η·n/d` and `n` comes from `solve_matched_ttt` reading the **CLU cell's byte ledger**. Pilot `(k,n)=(2197,52)`, `d=12` ⇒ **3.47 on 100 % of chunks**, ‖W‖ ×7.9e4 in one forward, NaN at step 135/4000; toy `(571,9)`, `d=3` ⇒ 2.31 on 44 %, bounded. ⭐ **Reproduces in 33 s at pilot *memory geometry* with a toy shell (NaN at step 107) — the criterion is geometry-only, so `d_model`/`n_layers`/`batch` are irrelevant to it.** Fix built and gated OFF: `PilotConfig.ttt_normalized_write` (normalized delta rule; scale-free; zero param/state cost). ⛔ **Default flip is an open Hub ruling — it also makes the rival stronger (2.12 vs 4.75 best-before-NaN).**
- **7.31 [FIXED on the same branch] Adding a field to `StreamMemoryConfig` invalidated every crash journal on disk.** The fingerprint's `memory` group is a full `asdict` while `pilot`/`store` are non-default-only tables. The four C2W6 fields (`ffe7440`) had **already stranded the banked CSF3 tier-iii journals on `main`** — verified: `load_journal` refused the real `pilot_pilot_seed0_PARTIAL.json`. Now reconciled when the post-dating key sits at its own field default (strict otherwise). ⚠ Any new `StreamMemoryConfig` field must ship OFF-by-default and bit-identical for this to stay sound; the durable fix is an `as_flag_table()` on that dataclass.
- **7.32 [standing, discipline] A toy bit-identity gate cannot certify the TTT arm at another scale.** Its stability criterion is a function of the *solved geometry*, which the toy does not share, and the toy sits *astride* the boundary (2.08–2.31, ~40 % of chunks amplifying) rather than safely inside it. Any TTT change must additionally run the 33 s **pilot-memory-geometry** rig, which is positive-control-verified (it NaNs on the shipped code).
- **7.33 [standing, launch checklist] A pre-registered phase gated behind a CLI flag no launch path sets is indistinguishable, in the artifact, from a deliberate cut.** D5 was in this state on *every* attempt. `job_gpu_cluformer.sh` now has a `D5` passthrough and a test asserts a passthrough exists for all seven flags; the checklist item is *assert every pre-registered phase appears in `stages_reached`/the phase list*.

**§3 (CLI & config) — new knob:** `PilotConfig.ttt_normalized_write: bool = False` (tier-iii pilot; `--set ttt_normalized_write=true`; **claims-relevant, default flip pending a ruling**).

**§10 — for the `2026-08-06` block:**
- Defect 1 **DIAGNOSED + LEVER BUILT, DEFAULT NOT FLIPPED (Hub ruling owed)**; defect 2 **WIRED**.
- ⭐⭐ **"Can D5 be obtained by re-resuming a finished leg?" — YES, and it is ~free: 1 phase runs, 22 are lifted** (replayed against the real journal; executed end-to-end at toy). `dyneval` — 219.09 GB, 61 686 s — is **lifted, not re-run**. Projected D5 host cost **≈ 0.02 GB**, i.e. it does not move the run peak and a D5-only pass fits `-G 1 -c 12`; wall **≤ ~1 h/leg** vs ~22 h for a retrain. ⛔ Preconditions: all five `.eqx` present; a checkout with **both** fixes; **no `ARMS` override**; back up the `S4.json` first.
- ⛔⛔ **New first-order fact: the banked journals were ALREADY unresumable on `main`** (four C2W6 `StreamMemoryConfig` fields) — the recovery path the checkpoint work bought had been silently closed since C2W6. Fixed on this branch; verified against the real artifact.
- **Never-quote candidate:** *"the toy bit-identity gates protect the scale run"* — measured false for the TTT arm (7.32).
