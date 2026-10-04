# WAVE 8 — THE GPU LEG, FOLDED INTO THE RECORD
**Compiled by the ICLR-long Hub, 2026-09-21.** Artefacts: `.claude/outputs/w8-encoder-loop/cloud/`.
⚠ Every number below is read from an agent's or a job's own manifest. ⛔ The Hub re-derived nothing.

## 1. What ran, and what it cost
**All three seeds COMPLETE**, A100-40 **spot**, project `forgisprova`, one attempt each,
⭐ **no pre-emptions and no stockouts.** Image `sha256:69a72b67…`, commit `b405f0d`, identical on
every seed. Sweep wall clock 2160 / 2152 / 2159 s.

**COST — measured, not estimated:** s0 **$1.02** · s1 **$1.02** · s2 **$1.25** = ⭐ **$3.30**,
plus **$0.57** lost to the earlier staging failure = ⭐ **$3.87 all-in**, at $1.470/h list.
⛔ **The $26.46 pre-estimate assumed the full 6 h `--max-hours`; the jobs took ~0.6 h.**

⛔⛔ **THE PINS ARE NOT WAVE 8's.** This leg ran `joint_steps=4000, dec_refit_steps=1800,
warm_steps=4000, n_train=32768`. The CPU leg that produced wave 8's table ran
**`400 / 400 / 1500 / 8192`**. The module's own rule: *"a run at one budget and a run at another do
not share a table."* ⭐ **So this is a SEPARATE TABLE. It can corroborate or contradict the direction
of wave 8's finding; it cannot add rows to it.**

## 2. ⛔ BAR D IS NOT YET SCORED — and it is not what the probe measures
⚠ **A correction the Hub records against itself:** BAR D is **not** the probe margin. Per the PREREG,
**BAR D = `pca_tensor@X − rival_mlp_180k@X` computed WITHIN each arm X, and whether that margin GROWS
from `alone` to `loop_swap`.** ⭐ If it does not grow, the features are better **for everyone** — a
finding about the encoder, ⛔ not about the store.

The cloud deliberately runs only `warm` and `train`; every manifest says
`stages_that_must_run_on_cpu: ["cache","probe","rivals","score"]`, because the feature-cache writer
refuses unless JAX sees exactly one CPU device. **`cache` and `probe` are DONE for all 3 seeds on the
Mac. `rivals` + `score` + the cross-arm report are NOT RUN** — a **DECLARED NOT-RUN**, not a null.

## 3. ⭐ WHAT THE GPU LEG DOES ALREADY SETTLE

### BAR C.1 — equal compute: **PASS on all three seeds**
`gradient_steps` **4000** and `images_seen` **256,000**, identical across `alone` / `loop_nosw` /
`loop_swap`, every seed.

### ⛔ BAR C.2 — the equal-SECONDS control: **FAILS on all three seeds**
| seed | `alone_wc` | `loop_swap` | shortfall |
|---|---|---|---|
| 0 | 297.6 s | 338.8 s | **−41.2 s** |
| 1 | 312.8 s | 340.6 s | **−27.8 s** |
| 2 | 318.2 s | 336.8 s | **−18.6 s** |

**The PREREG requires `alone_wc.train_seconds ≥ loop_swap.train_seconds`. It is short every time.**
⭐ **Cause, from the job log:** `alone_wc` sizes its step count from **`alone`'s** measured
seconds/step (*"matching loop_swap's 338.8s at alone's 0.0531s/step → 6376 steps"*), and then runs
**faster per step than that estimate**. ⛔ **Consequence: the equal-seconds control received 6–12 %
LESS compute than the treatment, so it is biased IN THE TREATMENT'S FAVOUR.** Any advantage
`loop_swap` shows over `alone_wc` on this leg is **not** a clean equal-seconds result and ⛔ must not
be reported as one. ⭐ The fix is one line — size `alone_wc` from its **own** seconds/step, measured
on a short pilot — and it costs nothing.

### BAR E — the probe, as a MARGIN over random init (⛔ no absolute is a claim)
| seed | `alone` | `loop_nosw` | `loop_swap` | A′ (nosw−alone) | A″ (swap−nosw) |
|---|---|---|---|---|---|
| 0 | −0.0505 | 0.0161 | 0.0261 | +0.0666 | **+0.0100** |
| 1 | +0.0242 | 0.0176 | 0.0304 | −0.0066 | **+0.0129** |
| 2 | −0.0862 | 0.0246 | 0.0321 | +0.1108 | **+0.0075** |

⭐ **The one consistent effect is the SWAP LOSS: A″ is positive on all three seeds** (+0.0100,
+0.0129, +0.0075). ⭐ **This corroborates wave 8's CPU decomposition** — *the swap loss is a real
ingredient* — at 10× the steps and 4× the data, on a different machine.
⛔ **A′ is NOT consistent** (+0.067, −0.007, +0.111) and ⛔ must not be quoted as a positive result:
its sign is driven by `alone`'s own instability, which is the next row.

### ⚠ A defect this leg exposes in the `alone` control
`alone`'s probe margin is **negative on two of three seeds** (−0.0505, −0.0862) — i.e. the TRAINED
encoder probes **worse than an untrained one** of identical architecture. The random-init floor is
constant within a seed (0.9714 / 0.9681 / …), so this is `alone`, not the instrument.
⚠ **This is the same instability the w9 harness independently found in its own encoder** (seed 1
stalled at BCE 0.080 where seeds 0,2 hit 0.005). ⭐ **Two waves, two codebases, the same failure mode:
a single training run of this encoder is not reliable.** ⛔ Until `alone` is stabilised, **every A′
column in the programme is measuring the control's variance, not the treatment's effect.**

## 4. What it would take to finish BAR D
`rivals` (2 arms × 3 seeds, `w4_learners` protocol verbatim) then the `test` score leg
(3 arms × 3 seeds) then the cross-arm report. ⛔ **Two blockers, both real:**
1. **It is a SCORED RUN.** `score.sh` refuses `test` without a PREREG **and** a row in
   `/c/chlu/SCORED-LEDGER.md` — a guard that is **deliberately not inheritable** from another hub.
   A new PREREG is needed because the pins differ. `encoder-loop` has **2 runs left**.
2. **The ledger is on the ThinkPad and the feature caches are on the Mac** — 1.8 GB per arm,
   **28 GB total**, so they are not shippable. The arms themselves are only **104 MB/seed**;
   regenerating the nine caches on the box costs ~25 min of CPU.
⛔ **The Hub has NOT fabricated a ledger on the Mac to get past the guard.** That is exactly the
stray-ledger failure in the policy's mistakes register.

⚠ **Priority note for the Head:** this leg *"answers BAR D only — it cannot overturn BAR A, since
both sides share the renderer."* **BAR D closes a LIMITATION, not a verdict.** It is therefore
ranked below wave 10's draft, and the Hub has not pre-empted that ordering.

---

## 5. ⚠ ARTEFACT LOCATION — changed 2026-09-21, record this before re-running anything
The GPU leg's working tree is **28 GB** (feature caches at 1.8 GB per arm). It was fetched into
`.claude/outputs/`, which `fork_push_runbook.sh` **backs up** — 478 MB of it in sub-90 MB files that
would have been staged into git, against the programme's own *"no large binaries in git, ever."*

⭐ **Moved to `.claude/scratch/w8-gpu/out/`** — declared regenerable and excluded from backup.
⭐ **The record is preserved:** every manifest this report cites (`arm.json`, `probe.json`,
`cache.json`, `cloud_unit.json`, `encoder_loop_report.json` — 46 files, 312 KB) is kept at
`.claude/outputs/w8-encoder-loop/cloud/manifests/`. **Every number in §3 traces to those.**
⚠ **Consequence for the scoring leg:** its run-base is now
`.claude/scratch/w8-gpu/out` — pass that as `<run-base>`. ⭐ The caches regenerate in ~162 s/arm and
the arms are still in `gs://chlu-cloud-forgisprova/jobs/w8enc-s{0,1,2}/`, so nothing is lost.
