# c3-rival-mamba2 — experiment-engineer report

**Task + acceptance criterion:** build the **Mamba-2** tuned rival arm (task §0–§2 + §A) so it trains a smoke leg end-to-end on the real stream and emits a byte-ledger row reproducing `RIVAL_SPECS["mamba2"]`'s **6,475,776 B to the byte**, every state-bearing hyperparameter provenanced, `shrink_to_budget()`'s solved knob used and stated, zero ladder arms trained.
**Status: done.**

⛳ **RECONCILIATION LIST (owner needed — protocol §5 corollary, in the first 10 lines).** Two items, both outside my ownership:
1. ⛔ **`chlu/eval/byte_ledger.py::ArmLedger.dtype_bytes` is hardcoded to fp32 and mis-states my arm's row** (it prints `dtype_bytes: 4` and "dtype = float32 (4 B/elt)" for a row whose bytes were computed at bf16). The **total is correct**; the declared width is not. One-line fix, **owner = `c3-gb-landing`** (§F1 below).
2. ⚠ **The shrink knob is solved on the rival's 24-layer reference geometry, but the shell deploys 12 layers** ⇒ the arm sits at **0.4949×** of the ceiling. Not re-solved by hand (task §1.4 forbids it). **Owner = the Hub / `PREREG-C3-LADDER.md`** (§F2).

---

## §0 DIAL DECLARATION (echoed before the first result)

- **Dial: none — rival implementation.** ⛔ No CLU number and no comparison appear below. I built a rival arm and showed it trains and ledgers. The comparison is the ladder's.
- **Laundering control:** the **anti-hobbling rule** is the governing discipline. This arm gets the official block's *every* branch (conv, gate, D-skip, RMSNorm) at official defaults; the two shell-imposed constraints are named, not hidden (§2.2).
- **Falsifies:** state bytes disagreeing with `RIVAL_SPECS`; a config inherited from a library default; an arm that cannot enter through the byte ledger. **None occurred.**
- **Does NOT falsify:** Mamba-2 beating the CLU. ⛔ Nothing here was tuned down; three smoke bpc numbers exist and **none is quotable in either direction** (not a claim venue).

## §1 What I did

1. **New module `chlu/eval/rivals/c3_mamba2.py`** — the **language-model ladder arm**: the official `mamba_ssm/modules/mamba2.py` **inference recurrence** as a stream-block memory cell. in-proj → split `(z, xBC, dt)` → causal depthwise **conv branch** + SiLU → split `(x, B, C)` → `Δ=softplus(dt+dt_bias)`, `a=exp(Δ·A)`, `A=−exp(A_log)` → `h ← a·h + (Δx)Bᵀ` → `y = hC + Dx` → **gated RMSNorm** (`norm_before_gate=False`, `group_size=d_inner/ngroups`) → out-proj. Init from the reference: `A~U(1,16)`, `Δ~exp(U(log 1e−3, log 1e−1))` floored at 1e−4 then inverse-softplussed, `D=1`, torch-default (=Equinox-default) linear/conv init.
   ⛔ **This is not `chlu/eval/rivals/mamba2.py`** — that is B′'s memory-gym arm, deliberately conv-less and gate-less. Both now exist and the docstring says which is which, in both directions.
2. **New module `chlu/eval/rivals/c3_registry.py`** — the generic C3 rival seam (`register_c3_rival`, `is_c3_rival`, `make_c3_rival_cell`, `c3_rival_deployed_row`). **The only shared surface the other two rival spokes need is one line in `C3_RIVAL_MODULES`.**
3. **Two rival-agnostic hooks** in `chlu/training/train_cluformer.py` (`solve_arms` builds the rival cell from *its own* pinned config and puts it in `swap_ledger`; `build_arm` dispatches on `ArmSpec.rival_cfg`) + `PilotConfig.rival` and `--rival ARM.KEY=VALUE`.
4. **`scripts/smoke_c3_mamba2.sh`** — the §1.6 smoke leg on **real enwik8**: train → checkpoint → hard-kill → **resume** → evaluate → dyn-eval → **retention slices** → **byte ledger**, with the byte-exact reproduction re-asserted from the shipped artifact.
5. **`tests/test_c3_rival_mamba2.py`** — 19 cases (§3).
6. **`PREREG.md` written before the module existed** (`.claude/outputs/c3-rival-mamba2/PREREG.md`), committing to every byte count, the occupancy, and the parameter count in advance.

⛔ **Zero ladder arms trained.** Everything ran at `--scale toy` (d_model 32, 2 layers, 3 steps, 0.6 MB of enwik8) or in unit tests.

## §2 The pinned arm

### 2.1 Config, and where every number comes from

`d_model 512 · n_layers 24 (reference) · d_conv 4 · expand 2 · headdim 64 · ngroups 1 · dtype_bytes 2 · conv_bias True · proj_bias False · rmsnorm True · use_D True · d_state 128→**39**`

**All 13 provenance strings begin `OFFICIAL IMPLEMENTATION:` — none begins `PAPER:`.** That is deliberate and is now a test: arXiv:2405.21060's per-size appendix was **NOT OBTAINED** by the scout (PDF would not parse; ar5iv front matter only), and I have **no web tool in this kit and no local copy** (`docs/` holds only 2605.14685 and our own PDFs) — so the appendix **remains NOT OBTAINED** and no reconciliation with it was possible. A `PAPER:` prefix on this arm would be a fabricated citation.

⛔ **No library default is inherited-and-then-claimed.** Each of `d_conv/expand/headdim/ngroups` is asserted equal to `RIVAL_SPECS`' pinned value in a test; `d_state` is the **only** number that differs from the official default, and only because `shrink_to_budget` solved it **down**.

### 2.2 Declared deviations (both structural to the swap, both identical for every arm)

1. **The seam is the shell's.** `StreamBlock` hands every cell a pooled per-**chunk** latent of width `dim = addr_dim+payload_dim` and takes the same width back, so the arm's in/out projections are `dim → d_in_proj` and `d_inner → dim`, and one "token" of the recurrence is one chunk. Everything *inside* the cell is Mamba-2's own geometry, at its own `d_model = 512` — ⛔ **not** inherited from the shell's `d_model`.
2. **Read-before-write.** The shell reads the state holding chunks `0..c−2` and then writes `c−1`. Mamba-2 publishes `y_t = C_tᵀh_t` (after the update); here it is `C_tᵀh_{t−1}`. The current chunk still reaches the read through the `D` skip and the `z` gate, as in the official block, so the cost is the current-step SSM term only — and the CLU arm pays the identical convention.

## §3 How I verified (commands + observed output)

### 3.1 The byte arithmetic — every PREREG number hit exactly

```
$ PYTHONPATH=… python -c "…c3_mamba2 / byte_ledger…"
published/layer elts 134912   total 6475776
RIVAL_SPECS          3237888          6475776     ← identical, to the byte
shrunk d_state 39 · per-layer 86484 B · total 24 L 2075616 B
deployed 12 L 1037808 B  occupancy 0.49487   within_budget True
deployed  2 L  172968 B
cell_ledger: params 87816 · state_floats 43242 · state_bytes 86484 · dtype_bytes 2
             state_breakdown {ssm_state 39936, conv_state 3306}
```
| PREREG §2/§3/§4 prediction | measured | verdict |
|---|---|---|
| published 24 L = **6,475,776 B** | 6,475,776 | ✅ to the byte |
| shrunk knob `d_state 128→39` | 39 (imported from `shrink_to_budget`) | ✅ |
| shrunk 24 L = **2,075,616 B**, occ **0.98973** | 2,075,616 / 0.989731 | ✅ |
| deployed 12 L = **1,037,808 B**, occ **0.4949** | 1,037,808 / 0.494865 | ✅ |
| deployed 2 L = **172,968 B** | 172,968 | ✅ |
| params/layer at seam dim 12 = **87,816** | 87,816 | ✅ |

### 3.2 The smoke leg — real enwik8, end to end

`bash scripts/smoke_c3_mamba2.sh` (three legs, ~6 min total, `PY=.venv/bin/python`):

```
✓ leg 1: the rival arm trained, banked and checkpointed   (hard-exit 137, ckpt_mamba2_seed0.eqx on disk)
✓ leg 2: resumed to a final artifact                       ([resume] arm 'mamba2': ckpt loaded, training SKIPPED)
✓ leg 3: slices added to a finished leg by resume alone
byte ledger: mamba2 172,968 B (0.08248x of 2,097,152 B), φ accounted
pinned  24 L @ d_state 128: 6,475,776 B = RIVAL_SPECS to the byte
shrunk  24 L @ d_state 39: 2,075,616 B (knob solved by shrink_to_budget, not by hand)
deployed 2 L: 172,968 B occupancy 0.0825
provenance: 13 pinned numbers, all PAPER:/OFFICIAL IMPLEMENTATION:
train: 3 finite steps; static+dyn-eval columns present
slices: 6/8 bins scored, n_scored 884
✅ MAMBA-2 SMOKE PASSED — trains, checkpoints, resumes, evaluates, slices, ledgers to the byte.
```
Artifact copy: `.claude/outputs/c3-rival-mamba2/smoke-ledger.json`. ⛔ The three bpc values in it are labelled `arms_bpc_NOT_QUOTABLE` in the JSON itself.

### 3.3 Unit tests — 19 passed

`pytest tests/test_c3_rival_mamba2.py -q` → **19 passed in 33.62 s**. What they actually pin:
- the **byte reproduction** and the **solved shrink** (not a hand number);
- **provenance**: every field of the pinned config has a string, all `OFFICIAL IMPLEMENTATION:`, and the geometry equals the official defaults except the solved knob;
- **the recurrence against an independent NumPy transcription** of `step()` (5 chunks, `atol 2e-5` on the read, the ssm state and the conv state) — the equivalence-shim discipline, not self-comparison;
- the conv state retains **exactly the `d_conv−1` raw past taps**;
- the state pytree is **exactly** what the ledger charges (`state_floats == ssm.size + conv.size`);
- the **learned initial state is PARAMETERS** (`init_state()` returns the parameter leaves themselves);
- **liveness** (the `ttt_matched` DEFECT-1 lesson): finite and bounded over a **200-chunk** stream at 3σ inputs (`‖ssm‖` ends at O(1), decay `exp(−Δ·e^{A_log}) ∈ (0,1)` is a contraction), and **every** cell parameter receives non-zero finite gradient — an inert rival would be a hobbled control;
- the **ladder seam**: the arm appears in `build_byte_ledger`, `within_budget`, φ accounted, the pinned reference table travels with the artifact, and `assert_shared_shell_identical` passes across `{clu_store, mamba2, none}` — **the shell is bit-identical, only `block.cell` changed**.

### 3.4 Full suite — **1800 passed, 0 failed**, against a named and re-verified HEAD

```
$ PYTHONPATH=/Users/user/Desktop/CHLU-wt2 .venv/bin/python -m pytest -q -p no:cacheprovider --no-cov
1800 passed, 36 warnings in 2880.03s (0:48:00)
```
- **Branch HEAD tested: `6464ae4`** (`git rev-parse HEAD` taken at launch and again at the end — unchanged).
- **Base HEAD re-verified: `main` = `0644c48` before the run and `0644c48` after it** — stable, so the green result is not invalidated by a concurrent merge (the "suite runs need a stable HEAD" hazard).
- **Count reconciles exactly:** the C3W1 merge banked **1781** at `f98f939`; **1781 + 19 new cases = 1800**. No pre-existing test changed status; the six `test_v1_hopfield_gate` warnings are the usual `Mean of empty slice`.
- ⚠ **Disclosure — process hygiene, no code impact.** Three orphaned pytest runs of *mine* accumulated (this harness kills a background wrapper at ~600 s but the `nohup`'d child survives), and I cleared them with `pkill -f "python -m pytest"`. **That pattern is unsafe with concurrent spokes**: a `c3-gb-landing` targeted run was active in `../CHLU-wt1` around the same window and I cannot prove I did not kill it (its run was re-observed alive 2 minutes later). The **1800-pass result above is from a single clean run started after that**, alone except for wt1's targeted subset. ⛔ Do not use a bare `pkill -f pytest` in this repo; match on the worktree path.

## §4 Findings

**F1 ⛔ (STOP-and-report; owner `c3-gb-landing`) — `byte_ledger.arm_ledger()` mis-declares the rival's element width.** `ArmLedger.dtype_bytes` defaults to `FP32_BYTES` and `arm_ledger()` never reads the width the cell declared, so my arm's row reads `dtype_bytes: 4` and `"dtype = float32 (4 B/elt)"` while its `total_state_bytes` was computed at **bf16**. The **total is right**; the declared width is wrong, and this module's own docstring says "a ledger that mixes them silently is a 2× lie". The correct number *is* in the artifact one level down (`swap_ledger.mamba2.dtype_bytes = 2`), so nothing is lost — but the top row must not say 4. Proposed one-liner, in `arm_ledger`:
```python
led = ArmLedger(..., dtype_bytes=int(row.get("dtype_bytes", FP32_BYTES)), ...)
#   and use that value in the `arithmetic` string instead of the FP32_BYTES literal
```
⛔ I did **not** make this edit (task §2: `byte_ledger.py` is not mine).

**F2 ⚠ — the solved shrink under-fills the envelope at the shell's layer count.** `shrink_to_budget` solves on the rival's own **24-layer** reference geometry (2,075,616 B = 0.98973×). The C3 shell is the scout's **attention-class 12-layer** reference, so the deployed arm holds 12 cell states = **1,037,808 B = 0.4949×**. Under an anti-hobbling reading the rival is entitled to the *largest* `d_state` that fits **as deployed**; under task §1.4 I may not re-solve. I therefore **stated it in every row** (`deployed` sub-dict: `deployed_n_layers`, `occupancy`, `reference_n_layers`) and leave the decision to the Hub/prereg. ⚠ The same arithmetic applies to **every** rival whose reference `n_layers` (24 for the recurrent class) differs from the shell's 12 — GDN-2 and TTT included; **SWA and TXL are already 12-layer references and are unaffected**.

**F3 ⚠ — a rival-ONLY ladder job cannot run today.** `exp_cluformer_pilot.run_pilot` indexes `models["clu_store"]` unconditionally in S1/S2 (`monitors_init`, `allocation_liveness_init`, `gradient_probe_init`, lines ~474–488), but `models` is built only from `pcfg.arms` and the CSF3 ladder submits **one arm per job** (`--arms "$ARM"`). Any job whose arm is not `clu_store` will `KeyError` there. This is **pre-existing** (it bites `gru_matched`/`ttt_matched` jobs identically) and outside my ownership; my smoke works around it by passing `--arms mamba2 clu_store`, and the workaround is documented in the script header rather than hidden.

**F4 (context, not a defect) — at *toy* the rival's state exceeds the CLU store's** (172,968 B vs 41,152 B, `state_vs_clu 8.41`), because the rival's geometry is its own fixed 512-wide reference while the toy store is tiny. At the ladder geometry the direction reverses (the CLU store is 5.52 MB = 2.63× the ceiling per `c3-csf3-harness` §5.1, the shrunk Mamba-2 is 1.04 MB). ⛔ Neither number is a comparison; the byte *match* is achieved by shrinking both sides to one ceiling, and the CLU side's shrink is not mine.

**F5 — the appendix reconciliation asked for in §A could not be attempted.** No web tool in this agent's kit and no local copy of arXiv:2405.21060. `NOT OBTAINED` stands exactly as the scout left it, and the module/test enforce that no string claims otherwise.

## §5 The sanity anchor — **there is none, and that is the point**

⭐ **Mamba-2 has no in-class published anchor on any Track-A venue**: the paper evaluates on MQAR synthetics, Pile scaling laws, zero-shot downstream and speed — **enwik8 and WikiText-103 do not appear** (scout §1.1/§1.1.1). The nearest published Mamba-2 numbers are **rival-vs-rival**: GDN-2's Table 2 gives Mamba-2 **Wiki ppl 16.79** and GDN-v1's Table 3 gives **16.56**, both at **1.3 B params / 100 B FineWeb-Edu tokens / 4 k context with a subword tokenizer** — a different corpus, tokenizer, parameter count and protocol on all four axes. ⛔ **Neither is a matched baseline and neither can sanity-check our implementation.** What sanity-checks this arm instead is *internal*: the byte-exact reproduction of the official cache formula, the independent NumPy transcription of `step()`, and the boundedness/gradient liveness tests — i.e. **structural equivalence, not a number**.

## §6 Flag-provenance table

| item | value |
|---|---|
| commit (branch tip) | `6464ae4` (`aa4092a`, `e9f1a76`, `6464ae4`) |
| base | local `main` `0644c48` (C3 head at spawn; ⚠ the Hub did not name a base in the spawn line — `c3-gb-landing` was 4 commits ahead in wt1 and unmerged, so I based on `main`) |
| seeds | `0` (smoke), PRNGKeys `0/1/3/5/7` in tests |
| scale | `toy`: `d_model 32, n_layers 2, seq_len 256, batch 2, steps 3, warmup 1, data_bytes 600000, eval/dyneval/slice_batches 2, slice_min_n 5, monitor_every 1, stop_after_arms 1 (leg 1 only)` |
| memory cfg | `chunk 32, address_steps 4, read_steps 4, traj_stride 2, psi_hidden 16, write_inner_steps 1, write_n_perturb 4` |
| corpus | **enwik8**, real, staged at `~/.cache/chlu/datasets/enwik8` (100 MB), `data_bytes=600000` |
| arms | `mamba2, clu_store` (⛔ `clu_store` present only because of F3) |
| rival config | `mamba2: d_model 512, n_layers 24, d_state 39 (solved), d_conv 4, expand 2, headdim 64, ngroups 1, dtype_bytes 2, conv_bias 1, proj_bias 0, rmsnorm 1, use_D 1` — **all defaults; `PilotConfig.rival` was left empty**, i.e. no override was used anywhere |
| budget | `MATCHED_STATE_BYTE_BUDGET = 2,097,152 B`, `enforce_state_byte_budget=True` |
| `ttt_normalized_write` | `False` (untouched default) |
| jax / backend | `0.9.0` / cpu (main venv, **not** a fresh worktree sync — w6 lesson) |
| env | `PYTHONPATH=/Users/user/Desktop/CHLU-wt2 /Users/user/Desktop/CHLU/.venv/bin/python` |

## §7 Git footprint

- **Branch** `agent/experiment-engineer/c3-rival-mamba2`, worktree `/Users/user/Desktop/CHLU-wt2`, base **local `main` `0644c48`** (unchanged before *and* after: re-verified — see §3.4).
- **Commits** (3, atomic, tagged):
  - `aa4092a` add the Mamba-2 C3 ladder arm, pinned to the official implementation
  - `e9f1a76` let a registered C3 rival enter the ladder through the byte ledger
  - `6464ae4` tests + smoke leg for the Mamba-2 arm: byte-exact, provenanced, live
- **Files touched — declared in full (task §2):**
  | file | new? | why |
  |---|---|---|
  | `chlu/eval/rivals/c3_mamba2.py` | **new** | mine: the arm |
  | `chlu/eval/rivals/c3_registry.py` | **new** | mine: the rival seam; ⚠ **the one file the other two rival spokes also need** (one line each in `C3_RIVAL_MODULES`) |
  | `tests/test_c3_rival_mamba2.py` | **new** | mine |
  | `scripts/smoke_c3_mamba2.sh` | **new** | mine |
  | `chlu/training/train_cluformer.py` | edit | ⚠ **shared**: +1 `ArmSpec` field, +1 import block, +9-line rival branch in `solve_arms`, +4-line deployed-row loop, +1 `if/else` in `build_arm`, +1 `PilotConfig.rival` field. All **rival-agnostic** so a second spoke's version conflicts textually, not semantically |
  | `chlu/experiments/exp_cluformer_pilot.py` | edit | ⚠ **shared**: `--rival ARM.KEY=VALUE` (argparse entry + 10-line parse) |
- ⛔ **Not touched:** `chlu/core/blocks.py`, `chlu/eval/byte_ledger.py` (both `c3-gb-landing`'s — F1 is reported, not fixed), the CLU arm, the corpora registry, `PREREG-C3-LADDER.md`, the other two rivals, `chlu/eval/rivals/{mamba2,deltanet,ttt,fit,ledger}.py`.
- `git diff --stat main..HEAD` = 6 files, **+1,231/−6** (the −6 are the `build_arm`/`solve_arms` lines re-indented into the new branches).
- Rebase onto base: **no-op** — `main` is still `0644c48`, verified from the main repo after the suite.
- **Branch ref verified from the MAIN repo** (protocol §3.2, the wave-4 lost-commits lesson): `git -C /Users/user/Desktop/CHLU log --oneline main..agent/experiment-engineer/c3-rival-mamba2` lists all three hashes.
- **Not pushed, no PR, not merged. Worktree `/Users/user/Desktop/CHLU-wt2` left in place** for review (it also still holds the smoke artifacts under its own `.claude/`; copies are in the main repo's `.claude/outputs/c3-rival-mamba2/`).

## §8 Open questions / risks

1. **F1 and F2 need owners** (see the reconciliation list). F2 in particular is a *prereg* question: does the matched-bytes control shrink each rival to the ceiling at its **own reference depth** (current behaviour, 0.49× as deployed) or at the **shell's** depth (0.99×)? The second is the anti-hobbling reading; I did not take it unilaterally.
2. **Three spokes, one registry.** `c3_registry.py` will be created independently by the GDN-2 and SWA spokes if they take the same route. Mine is deliberately generic: whoever merges second should keep one copy and add a line to `C3_RIVAL_MODULES`, not fork it.
3. **The chunk seam is the rival's real handicap and it is uniform.** Mamba-2 natively runs one recurrent step per *token* at `d_model`; here it runs one per *chunk* through a `dim`-wide bottleneck. Every arm has the same seam, so the comparison is fair — but a referee will ask, and the answer should be in the paper's method section, not discovered in review.
4. ⚠ **Operational, for the Head:** this harness kills a *background wrapper* at ~600 s while the `nohup`'d pytest child survives, so orphaned suite runs pile up and contend for CPU (three of mine did; the first "full suite" attempt was invalidated by that contention and re-run clean). Clearing them with a bare `pkill -f "python -m pytest"` can hit **another spoke's** run — see §3.4's disclosure. A per-worktree match (`pkill -f "PYTHONPATH=<worktree>"`) is the safe form.
5. **`n_head`/`ngroups` sensitivity is untested at the ladder scale.** The pinned `ngroups=1` is the official default; at `d_model 512` that gives `n_heads=16` sharing one `(B,C)` group. Nothing here measures whether the ladder-scale arm prefers more groups — and per §0 it is not my place to tune it.

## Proposed handover updates (for the Hub)

- **§7 new item (OPEN, owner `c3-gb-landing`):** `byte_ledger.arm_ledger()` hardcodes `dtype_bytes = FP32_BYTES` and ignores the width a cell declares in its `cell_ledger()`. Totals are correct; the row's declared width and its `arithmetic` string are wrong for any bf16-deployed rival (first observed on the `mamba2` arm: row says 4, cell says 2). One-line fix given in `c3-rival-mamba2.md` §4-F1.
- **§7 new item (OPEN, pre-existing, no owner):** `exp_cluformer_pilot.run_pilot` indexes `models["clu_store"]` in the S1/S2 phases while building `models` only from `pcfg.arms` ⇒ **the CSF3 ladder's one-arm-per-job submission cannot run any non-`clu_store` arm alone** (`--arms "$ARM"`, `job_gpu_c3_seeds.sh:165`). Affects `gru_matched`/`ttt_matched` equally; it is not new to the rivals.
- **§3 (config) addition:** `PilotConfig.rival: Dict[str, Any] = {}` (empty default ⇒ absent from `as_flag_table` ⇒ resume fingerprints unmoved) and the `--rival ARM.KEY=VALUE` CLI flag, for **declaring** a deviation from a rival's pinned config. Unknown keys raise.
- **§2 (architecture) addition:** `chlu/eval/rivals/c3_registry.py` (the C3 rival seam) and `chlu/eval/rivals/c3_mamba2.py` (the Mamba-2 **LM ladder** arm). ⚠ Note explicitly that `chlu/eval/rivals/mamba2.py` is a **different object** (B′'s memory-gym arm, no conv branch, no gate) so no one merges the two.
- **Prereg question for `PREREG-C3-LADDER.md` (F2):** at which depth is a rival's shrink knob solved — its own published reference depth (current: Mamba-2 lands at 0.4949× as deployed in a 12-layer shell) or the shell's? Affects every recurrent-class rival (Mamba-2, GDN-2, TTT), not SWA/TXL.
