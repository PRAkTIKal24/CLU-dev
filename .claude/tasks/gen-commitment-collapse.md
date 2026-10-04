# Task: gen-commitment-collapse — why 72 requests become 9 codes, and whether any knob undoes it

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-commitment-collapse.md` + artefacts under `.claude/outputs/gen-commitment-collapse/`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (all) · **`.claude/tasks/gen-clu-phase1-rulings.md` — R20 (the 30-bit design constant), R23.5, R27.1, R28 `gen-read-compose`, and R29 in full (it is the pivot that created this task)** · `.claude/outputs/gen-read-compose.md` (the measurement you are explaining).
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Base:** `main @ 5fee3f4` · **Branch:** `agent/experiment-engineer/gen-commitment-collapse` · **Worktree:** fresh (`../CHLU-gen-collapse`).
- **Env:** main venv (`PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python …`), ⛔ no `uv sync` in the worktree. ⛔ CSF3 is unreachable (2FA) — local CPU only.
- ⚠ **Budget ≈ 4 h.** ⭐ **Write per (config, seed) as it completes and skip any unit whose output file already exists** — a detached spoke keeps running but fails its LAST write. ⛔ Never hold results to the end.

## ⭐ DIAL DECLARATION
- **Dial:** *commitment* — how the request's posterior over modes becomes an integer code.
- **Laundering control:** ⭐ **the matched-seen corner (MS)**, on every single number. See "the control that can kill this" below.
- **Falsifies:** R29's pivot item 1. **PASS = ≥60/72 distinct codes on B AND the store→decoder arm within seed spread of `knn_proto` (in-corner shape balanced accuracy 0.8817 ± 0.0023).** ⛔ **Fail is a legitimate, reportable outcome and routes to new approaches, never to abandonment.** ⛔ Do not re-tune toward a pass.

## The one job
Measure the **distinct-code count on the 72 held-out requests** as a function of the three knobs R29 names — **stopping rule · multiplicity · launch temperature** — and say whether any setting reaches the pre-registered PASS.

## Where the collapse actually lives — start here, it will save you a day
`chlu/core/gen_compose.py::compose_codes` is a hard `argmax` over the per-well posterior, and
`chlu/core/gen_dream.py::address_requests` builds that posterior as a naive-Bayes sum of per-factor log-conditionals.
Montero's held-out cell is a **colour corner** (all three hues ≥ 0.8), so the three hue factors contribute the same
large log-conditional to every one of the 72 requests, and the shape/scale/orientation terms are not enough to move
the argmax. ⭐ **That is the mechanical hypothesis. Confirm or refute it before you sweep anything.**

## ⭐⭐ THE CONTROL THAT CAN KILL THIS — Hub-derived from disk, and it reframes the question
The collapse is **not** a property of held-out-ness. Distinct code configurations, from
`.claude/outputs/gen-read-compose/runs/shapes3d_recon/seed_*/compose_manifest.json`:

| set | what it is | seed 0 / 1 / 2 | distinct fraction |
|---|---|---|---|
| **A** | 512 seen requests, spread over the factor space | 494 / 493 / 498 of 512 | **≈ 0.97** |
| **MS** | ~198 matched-seen requests — **SEEN shapes, SAME colour corner** | 35 / 24 / 24 of ~198 | **≈ 0.14** |
| **B** | 72 held-out requests, same colour corner | 9 / 8 / 10 of 72 | **≈ 0.125** |

⭐ **MS and B collapse at the same rate; A does not collapse at all. The corner collapses, not the hold-out.**
⇒ **Every number you report carries its MS companion.** A knob that lifts B's distinct count while MS stays flat has
found a B-specific artefact, not a fix, and you must say so in those words. ⛔ A B-only improvement is not a pass.

## The four legs, in order
### 1. Diagnosis before sweeping — is it the posterior or the argmax?
Report, per set (A / B / MS) and per seed: **per-well posterior entropy**, **mean top-1 confidence**, and the
**margin between the top-1 and top-2 modes**. ⭐ **If the posterior is already degenerate (top-1 ≈ 1.0, near-zero
entropy), temperature and stopping rule CANNOT help** and the fix is upstream in `fit_address_book` / the grouping —
report that as the finding and do not burn the budget on a sweep that cannot move. Reference from disk: mean
posterior confidence on B = **0.856** (seed 0).

### 2. Sweep on the CHEAP statistic first
The distinct-code count needs **no decoder** — it is posterior → commitment → `np.unique`. Sweep all three knobs
broadly at that cost (seconds per config):
- **launch temperature** τ on the posterior before commitment;
- **stopping rule** — argmax (shipped) · sample · commit-only-above-a-margin with abstain;
- **multiplicity** — k > 1 modes per well, i.e. the k-particle launch actually launching k particles rather than 1.
  ⛔ **If multiplicity widens the channel beyond `K·log₂M = 30` bits, say so explicitly and report the realised width** —
  R20 made 30 bits a declared design constant, and any change to it is a declared change to the claim, not a free win.

### 3. Evaluate accuracy ONLY at a pre-registered shortlist
⛔ Do not decode every sweep cell. File in `PREREG.md`, **before** any accuracy run: the shipped argmax baseline plus
**at most one best setting per knob** (≤ 4 configs), chosen on the distinct-code statistic alone. Then run the full
`request → codes → read → frozen decoder → probe` path on those, 3 seeds.
⭐ **Both numbers or neither.** Distinct-count alone is trivially maximised by adding noise; the accuracy criterion is
the only thing that stops noise scoring as a win. ⛔ Never report a distinct count without its accuracy beside it.

### 4. ⭐ The rider that unblocks every future question — persist the codes
**Persist the per-request integer code tuples for A, B and MS, per seed**, at
`.claude/outputs/gen-commitment-collapse/codes_<set>_seed<k>.json`. ⭐ The Head asked whether the 9 codes coincide with
**seen** configurations and it could not be answered, because `gen-read-compose` persisted only the *count*. Persist
the tuples and answer it: **report |B-codes ∩ A-codes| and |B-codes ∩ MS-codes| per seed, one line each.**

## ⛔ Live traps — each has already cost this programme time
1. **`dec_recipe` defaults to `"baseline"`** — the decoder that FAILED the gate (0.7191). The winner is
   **`v1_capacity_schedule`** (0.9414). ⭐ **Select it explicitly and assert the decoder parameter hash.** A baseline
   run looks entirely normal in the logs and is **void**.
2. ⛔ **`soft_read(store, X, tau)` takes FEATURES.** Using it voids the structural no-bypass guarantee. Every mixture
   weight you compute comes from the **request** (`address_requests` log-scores), never from a feature tensor.
   **Assert your path touches no feature tensor**, exactly as `gen-read-compose` did.
3. ⚠ **Feature caches are non-uniform:** seed 0 is `scope="full"` (480,000 rows); seeds 1–2 are `scope="used"` and
   contain **none** of the matched-seen rows. Check before you index.
4. **JAX cold start ~20 min** here — a plain `import chlu` will time out a short tool call. Keep the session warm.
5. **16 GB machine, other spokes running.** One feature cache is 1.97 GB float16. ⛔ Never start a full test suite.

## Files you own
`chlu/core/gen_compose.py` · `chlu/experiments/exp_gen_compose.py` · `tests/test_gen_compose.py` ·
`chlu/core/gen_dream.py` — **`fit_address_book` and `address_requests` only** ·
`chlu/config.py` — ⭐ **ONE new config block, APPENDED AT END OF FILE. ⛔ Edit no existing block.**
⛔ Nothing under `chlu/core/gen_store.py`, `gen_bridge.py`, `gen_certificate.py`, `gen_request.py`,
`chlu/eval/rivals/`, `chlu/data/`, `scripts/csf3/`.
⚠ **`chlu/config.py` is shared with two concurrent spokes.** `gen-arms-a7-wiring` edits the existing
`GenBaselinesConfig` block; `gen-external-classifier` also appends at EOF. Your anchors do not overlap theirs.
⛔ **If you hit a conflict there, do NOT merge hunk-by-hunk** — take the peer's whole file, re-apply your block, and
assert **0 deletions**. Hunk-by-hunk "keep both" has broken this file's syntax twice.

## Acceptance
1. `PREREG.md` filed **before** the first accuracy run, naming the ≤4 shortlisted configs and the PASS criterion verbatim.
2. Diagnosis leg reported: posterior entropy / top-1 confidence / top-1−top-2 margin, per set, per seed.
3. Distinct-code sweep over all three knobs, **every row carrying its MS companion**.
4. Accuracy run on the shortlist, 3 seeds, against `knn_proto` **0.8817 ± 0.0023**; PASS/FAIL stated plainly.
5. Per-request code tuples persisted; the A∩B and MS∩B overlap counts reported, one line each.
6. Realised channel width reported for every config; any config exceeding 30 bits flagged as a declared change.
7. Per-unit incremental writes; any unit not completed reported as **NOT RUN**, never as a zero.
8. Targeted tests green. ⚠ **⛔ Do not start a full suite** — the Hub runs one at wave close.

## What would prove this spoke wrong
- ⭐ **A knob lifts B's distinct count and MS stays flat.** You have found a B-specific artefact. Say so.
- ⭐ **Distinct count rises and accuracy falls.** You have added noise, not commitment. That is the failure mode this
  task is shaped to catch, and reporting it is a result.
- The posterior turns out to be degenerate and you swept anyway. The diagnosis leg exists to stop that.
