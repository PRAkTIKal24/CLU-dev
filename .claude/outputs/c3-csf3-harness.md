# c3-csf3-harness — experiment-engineer report

**Task + acceptance criterion:** build the Track-A real-data harness on the merged pilot infra — corpus
registry, within-document retention slices, seeds-as-jobs resume-first ladders, the local smoke config,
and byte ledgers at the ruled matched-state-byte budget — and demonstrate it runs end-to-end.

**Status: done.** One deliberate **STOP** (§7.1: the `d_addr` probe's number is exposed as a documented
constant, *not* flipped into `PilotConfig.addr_dim`'s default — that is a Hub ruling, and doing it would
refuse every banked CSF3 journal) and one **first-order finding the Hub must act on before run 3**
(§5.1: at the pilot geometry the CLU arm is **2.63× over the ruled 2 MiB budget**).

> ⚠ **RECONCILIATION LIST — needs an owner (protocol §5 corollary, in the first 10 lines).**
> 1. ⛔ **The pilot config busts the ruled budget by 2.63×** (§5.1). The harness now refuses to train it.
>    Someone must decide: shrink the store, or re-rule the counting convention. **Blocks run 3.**
> 2. ⚠ **The per-layer-vs-total counting convention is genuinely ambiguous** (§5.2) and the scout's own
>    table mixes it with a bf16-vs-fp32 element width. One line from the Hub settles both.
> 3. ⛔ **`d_addr` default NOT flipped** (§7.1) — a Hub ruling, with the reason measured.
> 4. ⚠ **`MATCHED_STATE_BYTE_BUDGET = 2,097,152 B` is built against the recommendation**, since the
>    last-digit confirmation had not landed. One-line edit if it differs.

---

## ⭐ DIAL DECLARATION (protocol §7, echoed before the first result)

- **Dial / pillar:** **none — instrument/harness.** ⛔ I produce **no claim, no tier-iii number, no bpc a
  paper could quote, and no verdict about the CLU vs anything.** Every number below is either an
  arithmetic property of a config, a property of the *instrument*, or a smoke-scale execution trace.
- **Laundering control:** not mine to pass, mine to make **impossible to omit** — the dyn-eval substitute
  column and the per-arm byte ledger **including φ** are now structural outputs. The byte ledger is built
  *before training* and raises; the retention slice is computed for every arm **and for every arm's
  dyn-eval column** in the same call (`_arm_slices`), so an arm cannot be reported without them.
- **Falsifies the task:** smoke not running end-to-end; a resume across a config-field addition being
  refused; a ledger that cannot account for φ; a slice that changes value when only *content* is permuted.
  **All four were run and none fired** (§3, §4, §6).
- **Does NOT falsify the task:** the smoke numbers being bad, or any arm beating any other on the smoke
  config. ⛔ **The smoke config is never a claim venue** — the script says so in a banner at the top.

**Pre-registration:** ⛔ not required and none filed — the acceptance criterion is an instrument, not a
measured ratio/exponent/slope/law (protocol §5, and the task says so explicitly).

---

## 0. Mechanical precondition — VERIFIED before any worktree, any code

```
$ git cat-file -e main:tests/test_ttt_stability_and_d5_wiring.py && echo PRECONDITION-MET
PRECONDITION-MET
$ git log --oneline main | head -3
60d8be4 [hub] demote the toy compositional substrate to a smoke/regression instrument
5656728 [hub] merge pilot-ttt-nan-and-d5-wiring (C2 close-out): ... the journal guard repaired
c8314a8 [hub] merge c2w11-organizer-swap-nulls ...
$ git log --oneline main..agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring
(empty)
```

✅ All three checks pass. Branched off `main @ 60d8be4` in worktree `../CHLU-c3harness` (the shared
checkout was on the live `pilot-ttt-nan-and-d5-wiring` branch, so a worktree was mandatory).

**Non-blocking input — the `d_addr` ceiling probe: PRESENT.** `.claude/outputs/c2w11/DADDR-CEILING-PROBE.json`
(7,964 B, 2026-08-12 23:51). Its number and what I did and did **not** do with it: **§7.1**.

**Environment note (contradicts the standing §4 warning, in the good direction):** JAX cold start in this
worktree against the main venv was **15.1 s**, not ~20 min (`jax 0.9.0`, import 2.2 s; `exp_cluformer_pilot`
import 14.9 s). I reused the main venv per the w6 lesson (`PYTHONPATH=<worktree>
/Users/user/Desktop/CHLU/.venv/bin/python`) and never ran `uv sync` in the worktree, so **JAX stayed 0.9.0**.

---

## 1. Track-A data plumbing — the seam, not the loaders

**New: `chlu/data/corpora.py`.** A stream is now a **config value** (`PilotConfig.corpus`), not a code path.

| piece | what it does |
|---|---|
| `CorpusSpec` | `load` · `stage` · `vocab_size` · `level` · `metric` · **`doc_boundary`** · `citation` |
| `register_corpus` / `get_corpus` / `available_corpora` | the registry; an unknown name lists the registered ones |
| `load_corpus` / `stage_corpus` | the only two calls the trainer/experiment make |

**⭐ Corpus-GENERIC, not a two-way switch — and the claim is auditable, not asserted.** `SEAM_COST` in the
module states exactly what a third stream costs, and it is verified by construction:

1. a new `chlu/data/<corpus>.py` exposing `stage_*`/`load_*` with the `Enwik8Split` surface;
2. one `register_corpus(CorpusSpec(...))` call;
3. one config value `PilotConfig.corpus='<corpus>'`.

⛔ **Nothing in that list is a trainer, iterator, slice or ledger edit.** That is the FineWeb-Edu priced
option (Advisor decision 5) staying *genuinely* cheap. ⛔ **I built no FineWeb-Edu loader and no PG-19
loader.** PG-19's three scout caveats are banked verbatim in `SEAM_NOTES["pg19"]` (the ~28,752-file inode
hazard → one memmap-able uint8 stream + sha256; the **word-count normaliser taken from the raw text, not
the tokenizer**, which is why a byte model can be scored in PG-19's own currency; the 50-book/3.0 M-word
validation set ⇒ **report per-book spread, never the bare mean**), with the NO-GO-as-external-venue
verdict attached. A test asserts all three are present *and* that neither corpus is registered.

**⚠ Staging discipline is ENFORCED, not documented (task §1.2).** `in_array_task()` detects
`SLURM_ARRAY_TASK_ID`; inside an array task **downloads are refused unconditionally**, whatever the caller
passed, and `NotStagedError` prints the exact serial staging command. A runbook comment does not stop 15
tasks racing one 36 MB fetch; this does. The `STAGE_ONLY=1` path in the job script additionally refuses to
run *as* an array task.

**Staging smoke (task §1.3):** the cached path is used and the split boundaries are byte-exact — the
smoke run loads `enwik8` at `data_bytes=600000` → `train 540,000 / valid 30,000 / test 30,000` B, exactly
the canonical 90/5/5 positional proportions, from the pre-staged cache with `download` never reached.
⛔ I did **not** touch the split arithmetic or the vocab-from-train-only rule.

---

## 2. The eval slices — SPEC, then the build (task §2 asks for the spec in this section)

**New: `chlu/eval/text_slices.py`.** Verified before building that this did not exist (`grep -rln revisit
chlu/ tests/` → only the toy/gym-side files).

### 2.1 The borrowing and the extension, separable by a reader

- **ADOPTED — Sun, Krishna, Mattarella-Micke, Iyyer (2021), *Do Long-Range Language Models Actually Use
  Long-Range Context?*, EMNLP 2021, pp. 807–822, arXiv:2109.09115.** Their *definition* of the slice and
  their *bucket structure*: target positions bucketed by **distance to the last occurrence in the
  prefix**, plus the **"never appears in the prefix"** bucket. ⛔ Not renamed; cited in the module
  docstring and in every emitted artifact's `citation` block.
- **PRECEDENT for the control — Khandelwal, He, Qi, Jurafsky (2018), ACL, aclanthology.org/P18-1027** —
  perturbation-by-distance (effective context ≈200 tokens; word order matters only within ≈50).
- **DECLARED AS OURS**, one line each, in the module and in the artifact: **(1)** the bin edges;
  **(2)** computing it on **enwik8/WT-103 bytes** rather than PG-19 subwords; **(3)** computing it **for
  every arm including the dyn-eval column**; **(4)** the **shuffled-position control applied to the slice
  itself**.

### 2.2 The revisit unit — ⚠⚠ TRAP 1, and it is measured, not asserted

**Unit = the enclosing whitespace-delimited TOKEN, on a byte stream.** Every byte position inherits the
revisit distance of the word it sits inside; whitespace positions carry no unit (bin `-1`) and are
excluded from every bucket rather than diluting them with the easiest bytes in the corpus.

**Why:** at vocab 256 the raw-byte unit is a character-frequency count. **Measured on enwik8** (valid
split, 200 kB, `<page>` boundaries):

| unit | median revisit distance | mass below 32 B |
|---|---|---|
| **token (shipped)** | **614 B** | 3.6 % |
| raw byte (the degenerate unit) | **14 B** | **70.5 %** |

⇒ ratio **43.9×**. The scout's warning is exactly right and the effect is an order of magnitude.

**⛔ The degeneracy is a tripwire, not a design note.** `assert_non_degenerate()` recomputes the index at
the raw-byte unit and **raises** unless the token unit clears it by ≥10× *and* exceeds 64 B absolute. A
test proves the tripwire can actually fire (a degenerate stream raises `DEGENERATE REVISIT SLICE`) — a
check that cannot fail certifies nothing.

**Distance bins (ours), in BYTES**, because the model's context is denominated in bytes:
`[1,8) [8,32) [32,128) [128,512) [512,2048) [2048,8192) [8192,inf)` **+ `never`**.
They straddle the pilot's **1024-byte context** (512 and 2048 sit either side) and the top bin is **8× the
context** — the regime where only a memory can help. The two tightest bins are deliberately narrow so a
degenerate slice would pile into them *visibly* rather than hide in a coarse first bucket.
⛔ `never` is **not** merged into the largest distance bin: "never seen" and "seen 10 kB ago" are different
questions, and that bucket is where a memory either shows up or does not.

**Retention** = per-bin **bpc conditioned on distance-since-last-occurrence**, so a memory that retains
appears as a gap that **widens with distance** rather than as a scalar (`slice_gap()` emits the per-bin gap
with both `n`s).

### 2.3 Document boundaries — explicit and tested per corpus

"Within-document" is load-bearing: a revisit measured across a document join is a coincidence between two
unrelated articles, and it lands in exactly the long bins the claim is read off.

- **enwik8** — `b"<page>"` (raw MediaWiki XML; measured 268 `<page>` in the first 2 MB ⇒ ~7.5 kB/document,
  several times the 1024-byte context, so the long-horizon slice is physically present, not nominal).
- **WT-103** — `b"\n = "`, with the **level-2 rejection**: ` = = Section = = ` must not open a document.
  Tested: `\n = Alpha = \n…\n = = Sub = = \n…\n = Beta = \n` → starts `[0, 46]`, and `22` (the section) is
  asserted absent.
- A corpus declaring `doc_boundary=None` is **refused** (`ValueError`), not silently scored across joins.

### 2.4 The three validity controls — all executed on real enwik8

| control | requirement | measured |
|---|---|---|
| **(0) non-degeneracy** (TRAP 1) | token unit ≥10× the byte unit | **43.9×** (smoke run: 32.6×) ✅ |
| **(a) shuffled positions** — permute token order within document, content multiset preserved | the slice **MUST move** | median 614 → **1241 B**, **TVD 0.0585** ✅ |
| **(b) content relabel** — injective, length-preserving relabelling, distances preserved | the slice must **NOT move** | bin counts **identical**, injective ✅, documents 15 → 15 ✅ |

⚠ **Control (b) failed on my first implementation and the failure was real, not cosmetic** — my relabelling
was non-injective for short tokens *and* it destroyed the `<page>` markers, so documents changed and the
distances genuinely moved. Fixed by (i) a per-length base-92 injective code and (ii) leaving structural
tokens (containing `<`, `=`, `&`) verbatim while excluding those bytes from the replacement alphabet, so a
relabelled token **cannot forge a boundary**. I report this because the dial declaration names exactly this
outcome as a falsifier: it was an instrument bug, caught by the control, and the control is what caught it.

### 2.5 Emission

Per-bin **counts alongside per-bin bpc**, always. ⛔ A bin below `slice_min_n` reports `"bpc": None` **with
its `n`** and `"sufficient": false` — never silently averaged away (tested). Written as its own artifact
`slices_<scale>_seed<N>.json` with the citation block and the controls attached.

⛔ **No slice value from either real stream is reported here as a finding.** The smoke slice below exists
only to show the instrument runs.

---

## 3. Job ladders — seeds-as-jobs, resume-first

**New: `scripts/csf3/job_gpu_c3_seeds.sh`.** The unit of work is **one (arm, seed) pair = one array task =
one A100**. `5 arms × 3 seeds = 15 tasks from ONE sbatch`.

*Why not extend `job_gpu_cluformer.sh`:* it runs `--seeds 0 1 2` inside one job and loops all five arms
inside that. At the scout's costing (≈1.5 h at 35 % MFU to ≈18 h at 3 % MFU per arm) that is ~270 h against
the **96 h per-job limit**, and one failure loses everything. Charter §4 is explicit that multi-seed =
seeds-as-jobs. ⛔ I did not modify `job_gpu_cluformer.sh`; its **D5 passthrough is untouched and asserted
still present** by a test.

- **≥3 seeds is the easy number:** `N_SEEDS="${N_SEEDS:-3}"` and `#SBATCH -a 0-14%4` are the defaults; the
  whole ladder is **one documented command line**, inside the 2×A100 / 4-day envelope (`-t 1-00:00:00`,
  `%4` = the free-tier concurrency cap).
- **⛔ Each arm gets its own `--out`** (`$OUT_BASE/<arm>_s<seed>`): `arms` **is** a `PilotConfig` field, so a
  narrowed arm list is a *different* config and a shared directory's journal would be refused. This is the
  pilot report's option-(b) discipline applied mechanically.
- **The `.eqx` precondition check before any re-resume** (ruling (2) of the pilot's reconciliation list) is
  implemented **twice**: in the job script (refuses to submit, before the GPU is touched) and in
  `run_pilot` (`resume_require_ckpt`, default `True`) — a journal that banks an arm as trained whose `.eqx`
  has vanished now **fails loudly** instead of silently retraining ~16 h.
- **Passthrough completeness** for `RESUME/D5/SLICES/MEM/STORE/SET` + `--corpus`, tested — §7.33's rule that
  a pre-registered phase behind a flag no launch path sets is indistinguishable from a deliberate cut.
- **⚠ One literal command line per job.** The task's zsh warning **fired on me live**: a `$CFG`
  args-in-a-variable invocation silently produced a broken run during development. The scripts carry the
  warning at the call site and every launch is literal.

**Resume-first, demonstrated end-to-end (not asserted):**

| leg | what ran | wall |
|---|---|---|
| fresh run (2 arms, slices ON) | everything | **198 s** |
| `--resume` on the finished leg | **every phase lifted, nothing retrained** | **12 s** |
| artifact comparison | resumed vs uninterrupted | **0 differing leaves** |
| `--resume --slices` on a leg trained **without** slices | **only the 3 slice phases ran**, all else lifted | 172 s → **35 s** |

That last row is the ladder property: `--slices`, like `--d5`, is a **CLI argument and not a `PilotConfig`
field**, so it cannot enter `rec["flags"]` and **cannot move the resume fingerprint** — a finished leg
gains the instrument by re-resume and nothing else re-runs.

---

## 4. The smoke config — `scripts/smoke_c3_local.sh`

Real stream (enwik8), tiny shapes, **minutes on the laptop**, exercising the full acceptance path in three
legs, the first of which is a **genuinely interrupted run** (`stop_after_arms=1` → `os._exit(137)`, no
finalisers, exactly as an `oom_kill` has none) so the resume is gated by a real interruption rather than by
inspection.

```
✓ leg 1: journal + checkpoint on disk after a simulated kill
✓ leg 2: resumed to a final artifact
✓ leg 3: slices added to a finished leg by resume alone
byte ledger: budget 2,097,152 B, enforced, φ on every arm
   clu_store        41,152 B  occupancy 0.01962  within=True
   echo                  0 B  occupancy 0.00000  within=True
   gru_matched         336 B  occupancy 0.00016  within=True
   none                  0 B  occupancy 0.00000  within=True
   ttt_matched      41,216 B  occupancy 0.01965  within=True
slice controls: non-degeneracy ratio 32.6x (token vs raw-byte unit),
                shuffle TVD 0.0448, relabel invariant True
   clu_store    slices: 6/8 bins scored, n_scored 884, dyn-eval column present=True
   none         slices: 6/8 bins scored, n_scored 884, dyn-eval column present=True

✅ SMOKE PASSED — load, train, checkpoint, resume, eval, slices, ledger.
⛔ Reminder: these numbers are NOT a result. This is not a claim venue.
```

⛔ **"NEVER A CLAIM VENUE" is a banner in the first 15 lines of the script**, where an operator will read
it, and the assertion block repeats it on exit. A test asserts the banner is in the head of the file.
Note the run ledgers **all five arms** although only two ran — see §5.3.

Artifacts on disk: `.claude/outputs/c3-csf3-harness/smoke-run/` (`pilot_toy_seed0_S3.json`,
`slices_toy_seed0.json`, `pilot_toy_seed0_PARTIAL.json`, two `.eqx`).

---

## 5. Byte ledgers and the matched-state-byte budget

**New: `chlu/eval/byte_ledger.py`.** Structural: built **before any training**, for every arm, from the
config, with the arithmetic in the artifact. `UnledgeredArmError` and `StateByteBudgetError` both **fail
the run loudly**; neither warns. A **zero-state** arm (`none`, `echo`) is a *ledger*, not a missing one —
0 is asserted, never inferred from absence. **φ is ledgered explicitly on every arm** with both its params
and state columns (`phi_accounted: true`), because an omitted φ row is indistinguishable from a forgotten
one; φ's inference **state is 0 and that is a statement** (it is a feed-forward per-chunk map), while its
**params are identical across arms** — which is what makes the swap a swap (`assert_shared_shell_identical`).

**The budget is ONE named constant**, `MATCHED_STATE_BYTE_BUDGET = 2_097_152` (2 MiB), with the ruling and
its rationale in the docstring and `BUDGET_PROVENANCE` carried into every artifact. ⚠ Built against the
Hub's recommendation because the last-digit confirmation had not landed; **one-line edit** if it differs.
⛔ I invented no third value.

### 5.1 ⛔⛔ FINDING — the pilot geometry is **2.63× over** the ruled budget

Computed from the config (not guessed), at `PILOT` (`d_model 512, n_layers 12, addr_dim 8, payload_dim 4,
capacity 32, atoms_per_item 256` ⇒ `dim 12`, `n_atoms 8192`, `state_floats 115,072`/layer):

| arm | state B/layer | × n_layers | total | occupancy of 2 MiB |
|---|---|---|---|---|
| **clu_store** | 460,288 | ×12 | **5,523,456** | **2.63×** ⛔ |
| **ttt_matched** | 456,976 | ×12 | **5,483,712** | **2.61×** ⛔ |
| gru_matched | 916 | ×12 | 10,992 | 0.005× |
| none / echo | 0 | — | 0 | 0 |

**The harness now refuses to train this config**, before a single A100-hour is spent:

```
⛔ matched-state-byte budget violated — the tier-iii control is not byte-honest at this config,
   so the run stops before it trains:
    clu_store: 5,523,456 B = 2.63x the 2,097,152 B budget
    ttt_matched: 5,483,712 B = 2.61x the 2,097,152 B budget
```

⇒ **This blocks run 3 and it is not mine to resolve.** The store must shrink (via `capacity`,
`atoms_per_item`, `addr_dim+payload_dim`, or `n_layers`) or the convention must be re-ruled. ⭐ Note the
two-sided swap is *internally* byte-honest — CLU 5.52 MB vs TTT 5.48 MB is a **1.007× match**, which is the
property the ruling wanted; it is the *budget*, not the *match*, that is violated.

### 5.2 ⚠ The counting convention is genuinely ambiguous — one Hub line settles it

The scout's §1.5 anchor is **"CLU store d=12 = 1,966,080 B"**, and our pilot cell is **460,288 B/layer**.
These are different objects, and two independent conventions are in play:

1. **per-layer vs total.** A 12-layer model holds **12** cell states at inference. The scout's *rival* rows
   are summed over `n_L` (e.g. TTT-Linear `24 × 33,280 elements`), so **total** is the like-for-like
   column — but the scout's *CLU* row is a single store. Ledgering per-layer instead would put `clu_store`
   at 460,288 B = **0.22×**, comfortably inside the budget. I ledger **both** (`cell_state_bytes_per_layer`
   and `total_state_bytes`) and enforce on the **total**, which is what actually occupies memory.
2. **element width.** Our arms are **float32 (4 B)**; the scout's rival table is **bf16 (2 B)**. `dtype_bytes`
   is a declared field on every row rather than an assumption, so the two are never silently mixed.

⛔ I did **not** choose between these — choosing is choosing the budget, which the task forbids.

### 5.3 Every job carries the full table

The ladder submits one arm per job, so a per-job artifact carrying only its own arm could not be assembled
into the matched-bytes table. `solve_arms` solves every cell regardless, so the ledger covers **all five
arms in every artifact** — verified in the smoke, where a 2-arm run ledgers 5 arms.

### 5.4 ⚠⚠ TRAP 2 — rival configs pinned, with per-number provenance

⛔ No library default is inherited. Every rival's state-bearing hyperparameters are pinned in
`RIVAL_SPECS`, each carrying whether the number came from a **paper table** or an **official
implementation**. A test asserts each `provenance` starts with `PAPER:` or `OFFICIAL IMPLEMENTATION:`.

**The six pinned rivals reproduce the scout's §1.5 derived table exactly** (asserted as a test, so drift is
caught), and `shrink_to_budget()` solves the declared knob **down** — the defensible direction:

| rival | pinned config (provenance) | bytes (bf16) | occ. | shrink-to-match |
|---|---|---|---|---|
| **ttt_linear** | H=8, d_h=64, 24 L (**impl**: ttt-lm-pytorch) | 1,597,440 | 0.76× | fits |
| **gated_deltanet2** | H=4, d_k=d_v=128, 24 L (**paper**: arXiv:2605.22791) | 3,145,728 | 1.50× | `n_heads` 4→6 ⇒ 2,097,152 |
| **transformer_xl** | mem_len 512, 12 L (**paper**: arXiv:1901.02860) | 6,291,456 | 3.00× | `mem_len` 512→170 |
| **mamba2** | d_state 128, d_conv 4, expand 2, headdim 64 (**impl**; paper appendix NOT OBTAINED) | 6,475,776 | 3.09× | `d_state` 128→39 |
| **sliding_window** | w=512, 12 L (**paper**: arXiv:2004.05150) | 12,582,912 | 6.00× | `window` 512→85 |
| **ttt_mlp** | H=8, d_h=64, 24 L (**impl**) | 12,705,792 | 6.06× | `head_dim` 64→25 |

⛔ The GDN-2 row's provenance names the trap explicitly: the `flash-linear-attention` defaults
(`head_dim=256, num_heads=6, expand_v=2`) give **3× the paper's state**, and a ledger built on them would
have voided the byte match silently. Artifact: `.claude/outputs/c3-csf3-harness/rival-state-byte-table.json`.

---

## 6. Verification — commands and observed output

| check | result |
|---|---|
| **New tests** `tests/test_c3_csf3_harness.py` | ✅ **37 passed** (36 + the `d_addr` STOP case) |
| **Full suite** on the branch | see §6.1 |
| `ruff check chlu/ tests/ scripts/` | ✅ **All checks passed** |
| `bash -n` on both new scripts | ✅ clean; executable bits set |
| Smoke end-to-end (3 legs) | ✅ passed (§4) |
| Resume bitwise vs uninterrupted | ✅ **0 differing leaves** |
| Slice position alignment vs the **real** `contiguous_batches` | ✅ `split.data[positions] == targets` |

⚠ **`ruff format --check` is not a repo gate** and I did not reformat (it fails on `main`'s own files too);
the pilot report notes the same.

### 6.1 Full suite

```
PYTHONPATH=/Users/user/Desktop/CHLU-c3harness .venv/bin/python -m pytest -q -p no:cacheprovider --no-cov
1781 passed, 29 warnings in 2470.44s (0:41:10)
```

✅ **1781 passed / 0 failed**, against **HEAD `f98f939`**, re-verified: HEAD `f98f939` before **and** after
the run; `main` unmoved at `60d8be4` on both sides; working tree clean on both sides. (A green against a
stale base is not a green.)

**Arithmetic checked, not assumed.** A `--collect-only` in a clean detached worktree at `main @ 60d8be4`
collects **1744**. My branch runs **1781 = 1744 + 37**, i.e. exactly my new test file (36 cases + the
`d_addr` case) and nothing else. ⚠ The pilot report's "1445 at `80d7d4b`" is stale relative to `60d8be4`
(three Hub merges later); **1744 is the current baseline** and the Hub may want to record it.

⚠ **One failure occurred on the FIRST full-suite run and I fixed it rather than reporting it as green.**
`tests/test_pilot_checkpoint_resume.py::test_the_final_artifact_carries_no_journal_key` pins the final
artifact's **exact top-level key list**, and making the byte ledger a structural output adds exactly one
key (`byte_ledger`) to it. That is a *deliberate content-shape change required by the task* (§5), not
instrumentation leaking out of the PARTIAL — so the correct fix was to update the pin, with the reason
recorded beside it, and re-run the whole suite. ⛔ This is the one edit I made to a file outside my
declared ownership list; it is 8 lines (5 added, 3 changed) in that single assertion, and it is listed in
§9. The green above is the **post-fix** run.

---

## 7. STOPs, open questions, risks

### 7.1 ⛔ STOP — the `d_addr` ceiling probe: number recorded, default **NOT** flipped

The artifact **exists** and I read it (not a report, not a charter):
`.claude/outputs/c2w11/DADDR-CEILING-PROBE.json`.

> **`d_addr_where_optimistic_exact_set_exceeds_V1_bar = [12]`** — the optimistic exact-set bound clears
> V1's bar (`v1_bar = 0.0504`) first at **`addr_dim = 12`** (`optimistic_exact_set_at_ceiling = 0.1375`
> vs 0.0340 at d=8). The artifact's own `reading.statement` says this is **NECESSARY, not sufficient**.
> Its regression anchor reproduced the banked `assignment_ceiling` at d=4 exactly (`abs_delta = 0.0`).
> ⚠ `placement_headroom_ceiling_minus_identity` **falls to 0.0 at d=12** (0.085 at d=4).

⛔ **I did not change `PilotConfig.addr_dim`'s default (8), and this is deliberate.** Flipping it to 12
would:

1. **refuse every banked CSF3 journal.** `PILOT` sets `addr_dim=8` explicitly; if the *default* became 12,
   `as_flag_table()` (non-default keys only) would start emitting `addr_dim`, the resume fingerprint would
   change, and all five banked legs would be refused — the precise failure the pilot merge repaired, and
   an explicit falsifier of *this* task ("a resume across a config-field addition being refused");
2. **move the TTT arm's stability criterion.** `dim = addr_dim + payload_dim` feeds `solve_matched_ttt`,
   and the pilot measured the divergence product as `η·n/d` — a pure function of the solved geometry. A
   d_addr change silently re-rolls the rival's inner-loop stability;
3. **move every state-byte number in §5**, i.e. the tier-iii control itself.

That is a claims-relevant change to a published control — a Hub ruling, exactly like
`ttt_normalized_write`. **`addr_dim` remains a plain config flag** (`--set addr_dim=12`), which is what the
task's fallback branch prescribes. **Recommendation: expose, do not flip, until the Hub rules.**

### 7.2 Other open items

1. ⛔ **§5.1 blocks run 3.** The harness will refuse the current pilot config. Needs a decision.
2. ⚠ **§5.2's two conventions** (per-layer vs total; fp32 vs bf16) want one Hub line each.
3. ⚠ **`enforce_state_byte_budget` defaults to `True`**, which is a *behaviour change* for any pilot-scale
   run. It is what the ruling asks for ("a mismatch fails loudly"), it cannot fire at toy scale (occupancy
   0.02), and the escape hatch (`--set enforce_state_byte_budget=false`) records `enforced: false` in the
   artifact so a non-compliant run is always *declared*.
4. ⚠ **`resume_require_ckpt` defaults to `True`**, also a behaviour change, on the path where a journal and
   its checkpoints disagree. The silent path has no legitimate use.
5. 🔍 **Not done, out of scope:** no PG-19 loader, no FineWeb-Edu loader, no dyn-eval re-measurement at
   26–47 M (the scout's point that the published 0.94 bpc is at 277 M stands; the harness treats dyn-eval
   as an arm, and `evaluate_slices_dyneval` gives it the slice too, but **measuring it is a run, not a
   harness change**).
6. ⚠ **The slice's per-token NLL adds one extra eval pass per arm** (plus one dyn-eval pass). At smoke
   scale the slice phases were 3 of 35 s. At pilot scale this is untested and should be watched in the
   first ladder job's `[rss]`/wall lines — it is the same call shape as `static`/`dyneval`, which are
   themselves the measured 340 s / 61,686 s phases.
7. ⛔ **`chlu/core/blocks.py` was NOT touched** (the kill condition). Nothing in the slices or the ledger
   needed it: the ledger reads `cell_ledger()`, which already exists on every cell.

---

## 8. Flag-provenance table

Every number in this report comes from one of four rigs. **Commit `f98f939`** (branch tip), **JAX 0.9.0**
(main venv, never re-synced), **CPU / float32**, **macOS**.

| | (A) slice instrument | (B) byte ledger | (C) smoke run | (D) tests |
|---|---|---|---|---|
| seed | 0 | 0 | 0 | 0 (fixtures seeded) |
| corpus | `enwik8`, real, `n_bytes=4,000,000` | n/a (config arithmetic) | `enwik8`, real, `data_bytes=600,000` | synthetic + `enwik8` config |
| scale | valid/test split, 200 kB sampled | `toy` and `pilot` as shipped | `toy` + overrides below | `toy` |
| non-default `PilotConfig` | — | — | `d_model=32, n_layers=2, seq_len=256, batch=2, steps=3, warmup=1, eval_batches=2, dyneval_batches=2, slice_batches=2, slice_min_n=5, data_bytes=600000, monitor_every=1, arms=("clu_store","none")`; leg 1 adds `stop_after_arms=1` | defaults |
| non-default `memory` | — | — | `chunk=32, address_steps=4, read_steps=4, traj_stride=2, psi_hidden=16, write_inner_steps=1, write_n_perturb=4` | defaults |
| slice params | `edges=(1,8,32,128,512,2048,8192)`, unit `token`, `doc_boundary=b"<page>"` | — | same, `min_n=5` | same |
| budget | — | **`2,097,152 B`**, `enforce=True` | same | same |
| CLI | — | — | `--corpus enwik8 --slices [--resume]`, `--d5` OFF | — |
| `ttt_normalized_write` | — | **`False`** (shipped default; the Hub's ruling is still owed) | **`False`** | **`False`** |

⚠ **Provenance of the two externally-sourced constant sets:** the **budget** (2,097,152 B) is the Hub's
recommendation under the Advisor+Head "≈2 MB" ruling of 2026-08-13, *not* measured by me; the **rival
pinned configs** carry per-row provenance in §5.4 distinguishing a paper table from an official
implementation (Mamba-2's paper appendix was **NOT OBTAINED** by the scout, so that row is code-sourced and
says so).

⛔ **No number in this report is a claim.** The smoke bpc values are deliberately omitted from every table
above except as execution evidence.

---

## 9. Git footprint

**Branch `agent/experiment-engineer/c3-csf3-harness`**, off local `main @ 60d8be4`, in worktree
`../CHLU-c3harness`. Not pushed, no PR. Rebase onto `main`: no-op (base unmoved).
⚠ Per §3.5 I did **not** rebase onto `origin/main` (stale at `40c2f31`).

| commit | files | note |
|---|---|---|
| `3f984f3` | **new** `chlu/data/corpora.py`; `chlu/data/__init__.py` (+16) | the registry |
| `01cd882` | **new** `chlu/eval/byte_ledger.py` | the ruled budget + rival pins |
| `2c48f81` | `chlu/training/train_cluformer.py` | ⚠ **shared file — exact hunks below** |
| `360a8f3` | **new** `chlu/eval/text_slices.py` | the slices |
| `74a13e3` | `chlu/experiments/exp_cluformer_pilot.py` | wiring |
| `c475b25` | **new** `scripts/csf3/job_gpu_c3_seeds.sh`, **new** `scripts/smoke_c3_local.sh` | the ladder + smoke |
| `2e770c1` | **new** `tests/test_c3_csf3_harness.py` | 36 tests |
| `f98f939` | `chlu/experiments/exp_cluformer_pilot.py`, `tests/test_c3_csf3_harness.py` (+1 test), ⚠ `tests/test_pilot_checkpoint_resume.py` (**+5/−3, one assertion**) | `DADDR_CEILING_PROBE`; the artifact-shape pin |

**⚠ `chlu/training/train_cluformer.py` — the shared file, minimal hunks, exact locations** (it was just
merged from the pilot work, so this is listed line-by-line):

1. **+1 line at the import block** (after `from chlu.data.enwik8 import bits_per_character`) —
   `from chlu.eval.byte_ledger import MATCHED_STATE_BYTE_BUDGET`.
2. **+38 lines inside `PilotConfig`**, purely additive fields, all at behaviour-preserving defaults:
   after `data_root` → `corpus`, `corpus_level`, `data_download`, `state_byte_budget`,
   `enforce_state_byte_budget`, `slice_min_n`, `slice_batches`, `resume_require_ckpt`.
3. **+21 lines after `_eval_loss`** — the new `token_nll` / `eval_token_nll` pair.

⛔ **No existing line of that file was modified or deleted** — every hunk is an insertion. `chlu/config.py`,
the CLI, `chlu/utils/plotting.py`, `chlu/core/blocks.py`, `chlu/eval/rollout_diag.py` and every other
campaign's code are **untouched**. `scripts/csf3/job_gpu_cluformer.sh` is **untouched** (its D5 passthrough
is asserted intact by a test, per the ownership note).

Scratch: none left in the repo; all outputs under `.claude/outputs/c3-csf3-harness/`.

**Worktree-ref verification (protocol §3.2, the lost-8-commits precedent):**

```
$ git -C /Users/user/Desktop/CHLU log --oneline main..agent/experiment-engineer/c3-csf3-harness
f98f939  2e770c1  c475b25  74a13e3  360a8f3  2c48f81  01cd882  3f984f3      (8 commits)
$ git -C /Users/user/Desktop/CHLU diff --stat main..agent/experiment-engineer/c3-csf3-harness
 10 files changed, 2562 insertions(+), 7 deletions(-)
```

✅ All 8 commits are visible on the shared ref **from the main repo**, checked before the worktree was
removed. Temporary worktree `../CHLU-c3base` (detached at `main`, for the collect-only baseline) was
created and removed; `../CHLU-c3harness` is removed after this verification.

---

## Proposed handover updates (for the Hub)

**§3 (CLI & config) — new knobs**, all shipping at behaviour-preserving defaults so no banked journal moves:

- `PilotConfig.corpus: str = "enwik8"` / `corpus_level: str = "byte"` — ⭐ **the real stream is now a config
  value**; `--corpus enwik8|wikitext103`. A third stream = one loader module + one `register_corpus()` call.
- `PilotConfig.data_download: bool = True` — **forced `False` inside a Slurm array task**, unconditionally.
- `PilotConfig.state_byte_budget: int = 2_097_152` (the constant
  `chlu.eval.byte_ledger.MATCHED_STATE_BYTE_BUDGET`) + `enforce_state_byte_budget: bool = True`.
- `PilotConfig.slice_min_n: int = 30`, `slice_batches: int = 0` (0 ⇒ reuse `eval_batches`).
- `PilotConfig.resume_require_ckpt: bool = True` — ⭐ **resolves ruling (2)** of the pilot's reconciliation list.
- CLI: `--slices` (a CLI arg, **not** a config field — like `--d5` it cannot move the resume fingerprint).

**§7 — new entries**

- **7.34 [NEW, BLOCKS RUN 3] The pilot geometry is 2.63× over the ruled 2 MiB state-byte budget.**
  `clu_store` 5,523,456 B / `ttt_matched` 5,483,712 B total inference state (12 layers × ~460 kB), against
  `MATCHED_STATE_BYTE_BUDGET = 2,097,152 B`. The harness now **refuses to train** it before any A100-hour is
  spent. ⭐ The two-sided swap is internally byte-honest (CLU/TTT = **1.007×**); it is the *budget* that is
  busted. ⚠ Two counting conventions are unsettled and each flips the verdict: **per-layer vs total**
  (per-layer would be 0.22× and compliant) and **fp32 vs bf16** (ours is fp32; the scout's rival table is
  bf16). ⛔ Neither is the engineer's to choose. **Owner needed.**
- **7.35 [standing, instrument] The byte-level revisit unit degenerates by an order of magnitude —
  measured.** On enwik8, median revisit distance is **614 B at the whitespace-token unit vs 14 B at the raw
  byte unit (43.9×)**, and the byte unit puts **70.5 % of its mass below 32 B** — a character-frequency
  count, not retention. `chlu/eval/text_slices.py` ships the token unit and a tripwire that **raises**
  (`assert_non_degenerate`), with a test proving the tripwire can fire.
- **7.36 [standing, discipline] A validity control caught a real instrument bug in this wave.** The
  content-relabel control (distances preserved ⇒ the slice must not move) **failed on the first
  implementation** — the relabelling was non-injective for short tokens and destroyed the `<page>` document
  markers. Controls that are cheap enough to run every time are worth more than controls that are argued.
- **7.37 [confirmed, ops] The zsh no-word-splitting trap fired during development.** Building the pilot's
  argument list in a shell variable (`$CFG`) produced a silently broken invocation. Both new scripts carry
  one literal command line per job plus the warning at the call site, and the ladder tells the operator to
  `ls logs/… | wc -l` after launching.

**§6 (env facts):** ⚠ **JAX cold start is not always ~20 min** — in a fresh worktree reusing the **main**
venv, `import jax` + `import exp_cluformer_pilot` took **15.1 s** total. The pathological cold start seems
to be cache-state-dependent, not a constant; budget for it but do not assume it.

**§10 — for a C3W1 block:** the Track-A harness is built and demonstrated (registry · retention slices with
three executed controls · (arm × seed)-as-jobs ladder · 3-leg local smoke · structural byte ledgers incl. φ
at the ruled 2 MiB). **Two items need an owner before run 3: the 2.63× budget bust (7.34) and the `d_addr`
default (§7.1, deliberately not flipped).**
